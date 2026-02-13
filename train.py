#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from torch.nn import functional as F
import torchvision
from utils.loss_utils import l1_loss, l2_loss, ssim, get_cluster_centroids, cosine_similarity, similarity_loss, uniformity_loss
from utils.loss_utils import entropy_loss, contrastive_clustering_loss_fast
from utils.geometry_utils import depth_to_normal, depths_to_points
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from model.slot_attention_mem import Attention, PositionalEncoding
from utils.vis_utils import visualizer_ply, visualizer_rgb, visualizer_semantic, visualizer_slot
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(f"{checkpoint}/gaussians.pth")
        gaussians.restore_feature(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), initial=first_iter, total=opt.iterations, desc="Appearance Training")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=False)

        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        render_pkg["gt_image"] = gt_image

        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # instance feature training
        if iteration > opt.densify_until_iter:
            if gaussians.ins_optimizer is None:
                gaussians.training_setup_ins(opt)

            ins_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=True, render_rgb=False)

            # instance feature loss
            instance_feature = ins_pkg["render_ins_feature"]  # [D, H, W]
            render_pkg["render_ins_feature"] = instance_feature
            instance_feature_flat = instance_feature.reshape(opt.instance_feature_dim, -1).permute(1, 0)  # [N, D]
            
            D, H, W = instance_feature.shape
            
            # Load gt instance masks from the camera
            gt_masks = viewpoint_cam.get_instance_masks(instance_mask_dir=dataset.im_path, levels=['m', 'l'])

            gt_instance_masks = torch.stack([gt_masks['m'], gt_masks['l']], dim=0)
            gt_instance_masks = F.interpolate(gt_instance_masks.unsqueeze(0).float(), 
                                         size=(H, W), mode="nearest").squeeze(0)
            instance_mask_flat = gt_instance_masks.cuda().long().flatten(1, 2) # Flatten
            
            # Compute contrastive clustering loss based on instance assignments
            loss += opt.lambda_ins * contrastive_clustering_loss_fast(instance_feature_flat[:, :D//2], instance_mask_flat[0], normalize=True)
            loss += opt.lambda_ins * contrastive_clustering_loss_fast(instance_feature_flat[:, D//2:], instance_mask_flat[1], normalize=True)

        loss.backward()

        iter_end.record()

        # Gaussian Update
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Gaussian densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if gaussians.ins_optimizer is not None:
                    gaussians.ins_optimizer.step()
                    gaussians.ins_optimizer.zero_grad(set_to_none = True)
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + "/ckpt" + str(iteration), exist_ok=True)
                torch.save((gaussians.capture_feature(), iteration), scene.model_path + "/ckpt" + str(iteration) + "/gaussians.pth")

            # Visualization
            if iteration % 100 == 0:
                depth = render_pkg["depth"]
                depth_normal, _ = depth_to_normal(viewpoint_cam, depth, world_frame=True)
                render_pkg["depth_normals"] = depth_normal
                visualizer_rgb(render_pkg, iteration, scene.model_path)

    print("Gaussian Appearance Training Completed!")

    with torch.no_grad():
        save_dir = os.path.join(scene.model_path, "rendering_final")
        os.makedirs(save_dir, exist_ok=True)
        viewpoint_stack = scene.getTrainCameras().copy()
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        for viewpoint_cam in viewpoint_stack:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=False)
            ins_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=True, render_rgb=False)

            image = render_pkg["render"]
            depth = render_pkg["depth"]
            pts_world = depths_to_points(viewpoint_cam, depth, world_frame=True)
            pts_world = pts_world.reshape(image.shape[1], image.shape[2], 3).permute(2, 0, 1)

            # instance feature loss
            instance_feature = ins_pkg["render_ins_feature"]

            cam_name = viewpoint_cam.image_name.split('.')[0]
            torchvision.utils.save_image(image, os.path.join(save_dir, f"{cam_name}.png"))

            torch.save(
                {
                    "render": image.cpu(),                    # [3, H, W]
                    # "depth": depth.cpu(),                    # [H, W]
                    "render_pts_world": pts_world.cpu(),            # [3, H, W]
                    "render_ins_feature": instance_feature.cpu()  # [D, H, W]
                },
                os.path.join(save_dir, f"{cam_name}.pt")
            )

    print("All Rendering Saved!")



def training_semantic(dataset, opt, save_dir, checkpoint_iterations, checkpoint, encoder='clip'):
    rendering_dir = os.path.join(save_dir, "rendering_final")
    view_stack = sorted(
        f for f in os.listdir(rendering_dir)
        if f.endswith(".pt")
    )
    
    # Set up Attention model
    use_ins = opt.use_instance_feature
    use_rgb = opt.use_rgb
    use_geo = opt.use_geometry
    optimizer = None
    if opt.train_semantic:
        # instance feature to semantics
        Attn = Attention(ins_dim=opt.instance_feature_dim,
                         tgt_feat_dim=opt.target_feature_dim, 
                         num_slots=opt.slot_num, 
                         in_slot_dim=opt.instance_slot_dim, 
                         tgt_slot_dim=opt.target_slot_dim,
                         use_geo=use_geo,
                         use_rgb=use_rgb,
                         use_ins=use_ins
                         ).cuda()
        
        if checkpoint and os.path.exists(f"{checkpoint}/attn_module.pth"):
            Attn.load(checkpoint)

        optimizer = torch.optim.Adam(Attn.parameters(), lr=1e-3)

    first_iter = 1
    total_iterations = opt.semantic_iterations
    progress_bar = tqdm(range(first_iter - 1, total_iterations), initial=first_iter - 1, total=total_iterations, desc="Semantic Training")

    batchsize = 8192
    for iteration in range(first_iter, total_iterations + 1):
        with torch.no_grad():
            # select a random view and load
            view = np.random.choice(view_stack)
            name = view.split('.')[0]
            render_pkg = torch.load(os.path.join(rendering_dir, view))

            image = render_pkg["render"].permute(1, 2, 0).cuda()
            pts_map = render_pkg["render_pts_world"].permute(1, 2, 0).cuda()

            instance_feature = render_pkg["render_ins_feature"].permute(1, 2, 0).cuda()
            
            # Load gt image
            image_path = os.path.join(dataset.source_path, "images", f"{name}.jpg")
            gt_image = Attn.load_gt_image(image_path)
            render_pkg["gt_image"] = gt_image
            gt_image = gt_image.permute(1, 2, 0)
            H, W, C = gt_image.shape
            
            # Load target semantic feature map
            tgt_feature, valid_mask = Attn.load_target_feature(dataset.lf_path, name, H, W, encoder=encoder)
            render_pkg["tgt_feature"] = tgt_feature
            tgt_feature = tgt_feature.permute(1, 2, 0).cuda()

            # Sample pixels
            random_idx = torch.randint(0, H * W, [batchsize])

            rgb_sample = image.reshape(-1, 3)[random_idx]  ##TODO image or gt image???
            pts_sample = pts_map.reshape(-1, 3)[random_idx]
            ins_feature_sample = instance_feature.reshape(-1, instance_feature.shape[-1])[random_idx]  # [H*W, D]
            tgt_feature_sample = tgt_feature.reshape(-1, tgt_feature.shape[-1])[random_idx]
            valid_sample = valid_mask.reshape(-1)[random_idx]

        # Attention forward
        if use_rgb:
            feature_sample = Attn.rgb_embed(rgb_sample)

        if use_ins:
            feature_sample = torch.cat([feature_sample, ins_feature_sample], dim=-1)
        
        if use_geo:
            geo_feature_sample = Attn.PEn(pts_sample)
            feature_sample = torch.cat([feature_sample, geo_feature_sample], dim=-1)

        out_feature, updated_in_slots, updated_tgt_slots, attn_weights = Attn(feature_sample.float(), tgt_feature_sample.float(), pts_sample.float())

        # Reconstruction Regularization
        # RGB loss
        recon_rgb = out_feature['rgb']
        rgb_loss = l1_loss(recon_rgb, rgb_sample)
        loss = opt.lambda_rgb_recon * rgb_loss

        # Instance feature loss
        # if use_ins:
        #     recon_ins = out_feature['ins']
        #     ins_loss = l2_loss(recon_ins, ins_feature_sample)
        #     loss += opt.lambda_ins_recon * ins_loss

        # Semantic loss
        recon_semantic = out_feature['semantic']
        tgt_loss = cosine_similarity(recon_semantic[valid_sample], tgt_feature_sample[valid_sample])  
        loss += opt.lambda_tgt_recon * tgt_loss

        # Slot Regularization
        # Entropy loss: each pixel only focus one slot
        ent_loss = entropy_loss(attn_weights, eps=1e-8, reduction='mean')
        loss += opt.lambda_ent * ent_loss

        # Attention loss: all slots being used
        attn_loss = (1 - attn_weights.max(dim=0).values).mean()
        loss += opt.lambda_attn * attn_loss

        # Slot difference loss: all slots to be different from each other
        # in_sim_loss = similarity_loss(updated_in_slots)
        # tgt_sim_loss = similarity_loss(updated_tgt_slots)
        # in_sim_loss = uniformity_loss(updated_in_slots)
        # loss += opt.lambda_sim * (in_sim_loss)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

        # Slots Update
        with torch.no_grad():
            Attn.update_slots(updated_in_slots, updated_tgt_slots)
            Attn.add_attn_status(attn_weights)

            # Slot attention densification
            if opt.slot_densify and (iteration - 1) % 1000 == 0 and iteration < (total_iterations // 2) and iteration > 1000:
                Attn.densification_and_prune()

            # Log and Save
            ema_loss_for_log = loss.item()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(save_dir + "/ckpt_semantic" + str(iteration), exist_ok=True)
                Attn.save(save_dir + "/ckpt_semantic" + str(iteration))

            # Visualization
            if iteration % 500 == 0:
                visualizer_semantic(render_pkg, iteration, save_dir, Attn, use_rgb=use_rgb, use_geo=use_geo, use_ins=use_ins)

            if iteration % 1000 == 0:
                visualizer_slot(render_pkg, iteration, save_dir, Attn, use_rgb=use_rgb, use_geo=use_geo, use_ins=use_ins)
            
            if iteration % 5000 == 0 and False:
                gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
                if checkpoint:
                    (model_params, first_iter) = torch.load(f"{checkpoint}/gaussians.pth")
                    gaussians.restore_feature(model_params, opt)

                visualizer_ply(gaussians, iteration, save_dir, Attn, use_rgb=use_rgb, use_geo=use_geo, use_ins=use_ins)
                del gaussians
                
    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    os.makedirs(save_dir + "/ckpt_semantic" + str(iteration), exist_ok=True)
    Attn.save(save_dir + "/ckpt_semantic" + str(iteration))

    print("Gaussian Semantic Training Completed!")

        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000, 15_000, 30_000])
    parser.add_argument("--ckpt_path", type=str, default = None)
    parser.add_argument("--encoder", type=str, default = 'clip')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    dataset_args.im_path = os.path.join(dataset_args.im_path, args.encoder)
    dataset_args.lf_path = os.path.join(dataset_args.lf_path, args.encoder)

    opt_args.target_feature_dim = 512 if args.encoder == 'clip' else 768

    # training(dataset_args, opt_args, pipe_args, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.ckpt_path, args.debug_from)
    
    ckpt_path = f"{dataset_args.model_path}/ckpt30000"
    training_semantic(dataset_args, opt_args, dataset_args.model_path, [5_000, 10_000], ckpt_path, encoder=args.encoder)
