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

import torchvision
from utils.loss_utils import l1_loss, ssim, constrastive_clustering_loss, cosine_similarity, entropy_loss
from utils.geometry_utils import depth_to_normal
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.vis_utils import apply_depth_colormap, colormap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from model.slot_attention import Attention
from sklearn.decomposition import PCA
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
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_rgb(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    optimizer_attn = None
    if opt.train_semantic:
        attn_module = Attention(in_feat_dim=opt.instance_feature_dim + 3,  ##TODO check input feature dim
                                tgt_feat_dim=opt.target_feature_dim, 
                                num_slots=opt.slot_num, 
                                in_slot_dim=opt.instance_slot_dim, 
                                tgt_slot_dim=opt.target_slot_dim
                                ).cuda()
        optimizer_attn = torch.optim.Adam(attn_module.parameters(), lr=1e-3)    

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
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

        render_instance = False
        if iteration > 15000:
            render_instance = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=render_instance)

        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        render_pkg["gt_image"] = gt_image

        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # geometry loss
        if opt.use_geometry:
            rendered_normal = render_pkg["render_normals"]
            median_depth = render_pkg["median_depth"]
            
            median_depth_normal, _ = depth_to_normal(viewpoint_cam, median_depth, world_frame=True)
            median_depth_normal = median_depth_normal.permute(2, 0, 1)
            normal_loss = (1 - (rendered_normal * median_depth_normal).sum(dim=0)).mean()
            
            
            expected_depth = render_pkg["expected_depth"]
            expected_depth_normal = None
            if expected_depth is not None:
                expected_depth_normal, rendered_points = depth_to_normal(viewpoint_cam, expected_depth, world_frame=True)  #TODO rendered normal is in world frame or not?
                expected_depth_normal = expected_depth_normal.permute(2, 0, 1)
                expected_normal_loss = (1 - (rendered_normal * expected_depth_normal).sum(dim=0)).mean()

                normal_loss = 0.4 * expected_normal_loss + 0.6 * normal_loss

            loss += opt.lambda_normal * normal_loss

            render_pkg["median_depth_normal"] = median_depth_normal
            render_pkg["expected_depth_normal"] = expected_depth_normal

        # instance feature loss
        batchsize = 4096
        if opt.use_instance_feature:
            instance_feature = render_pkg["render_ins_feature"]  # [D, H, W]
            
            # Load gt instance masks from the camera
            gt_instance_masks = viewpoint_cam.get_instance_masks(instance_mask_dir=dataset.im_path)
            instance_mask_flat = gt_instance_masks.cuda().long().flatten() # Flatten

            # Sample a random set of pixels from the image for contrastive learning
            random_idx = torch.randint(0, viewpoint_cam.image_width * viewpoint_cam.image_height, [batchsize])
            instance_feature_flat = instance_feature.reshape(opt.instance_feature_dim, -1).permute(1, 0) # Reshape instance features to (num_pixels, feature_dim)
            instance_feature_sample = instance_feature_flat[random_idx]

            # Compute contrastive clustering loss based on instance assignments
            instance_loss = constrastive_clustering_loss(instance_feature_sample, instance_mask_flat[random_idx])
            loss += opt.lambda_ins * instance_loss

        # slot training
        if opt.train_semantic:
            tgt_feature, valid_mask = viewpoint_cam.get_target_feature(dataset.lf_path, feature_level=0) # Shape gt_lan_feat: [512, H, W] Shape language_feature_mask: [1, H, W]
            
            # Attention forward pass
            instance_feature = instance_feature.permute(1, 2, 0) # From[D=16, H=730, W=988] to [H=730, W=988, D=16]
            rgb = image.permute(1, 2, 0)          # From [C=3, H=730, W=988] to [H=730, W=988, C=3]

            feature = torch.cat([rgb, instance_feature], dim=-1)  # [H, W, C+D]

            # Load gt instance masks from the camera
            gt_instance_masks = viewpoint_cam.get_instance_masks(instance_mask_dir=dataset.im_path) # Shape: [H, W]
            instance_mask_flat = gt_instance_masks.cuda().long().flatten()                          # Shape: [H*W]

            # Sample a random set of pixels from the image for contrastive learning
            random_idx = torch.randint(0, viewpoint_cam.image_width * viewpoint_cam.image_height, [batchsize])     # Shape: [batchsize]

            tgt_feature_flat = tgt_feature.reshape(tgt_feature.shape[0], -1).permute(1, 0) # Reshape lang features to (num_pixels, feature_dim)
            tgt_feature_sample = tgt_feature_flat[random_idx]

            feature_sample = feature.reshape(-1, feature.shape[-1])[random_idx]  # [1, H*W, C+D]

            out_feature, updated_in_slots, updated_tgt_slots, attn_weights = attn_module.train(feature_sample.float(), tgt_feature_sample.float())

            cossim_loss = cosine_similarity(out_feature, tgt_feature_sample)  
            loss += opt.lambda_cossim * cossim_loss
    
            ent_loss = opt.lambda_ent * entropy_loss(attn_weights, eps=1e-8, reduction='mean')
            loss += ent_loss

        loss.backward()

        iter_end.record()

        # Gaussian Update
        with torch.no_grad():
            # update slots
            if opt.train_semantic:
                attn_module.update_slots(updated_in_slots, updated_tgt_slots)

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

            # Slot attention densification
            if opt.train_semantic and iteration > 15000:
                if iteration % 5000 == 0:
                    pass  ##TODO slot attention densification

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if optimizer_attn is not None:
                    optimizer_attn.step()
                    optimizer_attn.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture_rgb(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Visualization
            if iteration % 1000 == 0:
                visualizer(render_pkg, iteration, scene.model_path, attn_module)


def visualizer(render_pkg, iteration, out_path, attn_module):
    gt_image = render_pkg["gt_image"]
    image = render_pkg["render"]

    depth = render_pkg["expected_depth"].squeeze(-1)

    rendered_normal = render_pkg["render_normals"]

    expected_depth_normal = render_pkg["expected_depth_normal"]
    median_depth_normal = render_pkg["median_depth_normal"]
    depth_normal = expected_depth_normal if expected_depth_normal is not None else median_depth_normal

    depth_map = apply_depth_colormap(depth[..., None], None, near_plane=0.1, far_plane=20)
    depth_map = depth_map.permute(2, 0, 1)
    
    _depth_normal = (depth_normal + 1.) / 2.
    _render_normal_world = (rendered_normal + 1) / 2

    instance_feature = render_pkg["render_ins_feature"]  # [D, H, W]
    D, H, W = instance_feature.shape
    x = instance_feature.permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x.cpu().numpy())  # [H*W, 3]
    feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
    feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min())

    instance_feature = instance_feature.permute(1, 2, 0) # From[D=16, H=730, W=988] to [H=730, W=988, D=16]
    rgb = image.permute(1, 2, 0)     
    cat_feature = torch.cat([rgb, instance_feature], dim=-1)  # [H, W, C+D]
    semantic_feature = attn_module.inference(cat_feature.reshape(-1, cat_feature.shape[-1]).float())  # [H*W, D]
    semantic_feature = semantic_feature.reshape(H, W, -1)
    x_pca = pca.fit_transform(semantic_feature.cpu().numpy())  # [H*W, 3]
    semantic_feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
    semantic_feature_vis = (semantic_feature_vis - semantic_feature_vis.min()) / (semantic_feature_vis.max() - semantic_feature_vis.min())

    row0 = torch.cat([gt_image, image, depth_map], dim=2)
    row1 = torch.cat([_render_normal_world, _render_normal_world, _depth_normal], dim=2)
    row2 = torch.cat([feature_vis, semantic_feature_vis, semantic_feature_vis], dim=2)

    image_to_show = torch.cat([row0, row1, row2], dim=1)
    image_to_show = torch.clamp(image_to_show, 0, 1)
    
    os.makedirs(f"{out_path}/log_images", exist_ok = True)
    torchvision.utils.save_image(image_to_show, f"{out_path}log_images/{iteration}.jpg")


def slot_visualizer(render_pkg, iteration, out_path, attn_module):
    os.makedirs(f"{out_path}/slot_visualization/{iteration}", exist_ok = True)

    gt_image = render_pkg["gt_image"]
    image = render_pkg["render"]
    instance_feature = render_pkg["render_ins_feature"]  # [D, H, W]

    D, H, W = instance_feature.shape

    instance_feature = instance_feature.permute(1, 2, 0) # From[D=16, H=730, W=988] to [H=730, W=988, D=16]
    rgb = image.permute(1, 2, 0)     
    cat_feature = torch.cat([rgb, instance_feature], dim=-1)  # [H, W, C+D]
    semantic_features = attn_module.per_slot_inference(cat_feature.reshape(-1, cat_feature.shape[-1]).float())  # [H*W, D]
    
    num_slots = len(semantic_features)
    for i in range(num_slots):
        feature = semantic_features[i].reshape(H, W, -1)
        pca = PCA(n_components=3)
        x_pca = pca.fit_transform(feature.cpu().numpy())  # [H*W, 3]
        feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
        feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min())
        masked_rgb = gt_image * feature_vis

        torchvision.utils.save_image(feature_vis, f"{out_path}/slot_visualization/{iteration}/slot_{i}_feature.jpg")
        torchvision.utils.save_image(masked_rgb, f"{out_path}/slot_visualization/{iteration}/slot_{i}_masked_rgb.jpg")
        

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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
