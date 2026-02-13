# copy from nerfstudio and 2DGS
import os
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.decomposition import PCA
from utils.sh_utils import SH2RGB
try:
    import open3d as o3d
except:
    o3d = None


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

# def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
#     """save points to point cloud using open3d"""
#     assert len(pts) > 0
#     if colors is not None:
#         assert colors.shape[1] == 3
#     assert pts.shape[1] == 3

#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(pts)
#     if colors is not None:
#         # Open3D assumes the color values are of float type and in range [0, 1]
#         if np.max(colors) > 1:
#             colors = colors / np.max(colors)
#         if BRG2RGB:
#             colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
#         cloud.colors = o3d.utility.Vector3dVector(colors)
#     if normals is not None:
#         cloud.normals = o3d.utility.Vector3dVector(normals)

#     o3d.io.write_point_cloud(path_save, cloud)
    

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    if img.shape[1:] != (H, W):
        img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
    return img


def save_ply(filename, points, colors):
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def visualizer_rgb(render_pkg, iteration, out_path):
    gt_image = render_pkg["gt_image"].cpu()
    image = render_pkg["render"].cpu()
    depth = render_pkg["depth"].squeeze().cpu()
    depth_normal = render_pkg["depth_normals"].permute(2, 0, 1).cpu()

    depth_map = apply_depth_colormap(depth[..., None], None, near_plane=0.1, far_plane=20)
    depth_map = depth_map.permute(2, 0, 1).cpu()
    normal_vis = (depth_normal + 1.) / 2.

    if render_pkg["render_ins_feature"] is not None:
        render_instance_feature = render_pkg["render_ins_feature"]
        D, H, W = render_instance_feature.shape
        x = render_instance_feature.permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
        pca = PCA(n_components=3)
        x_pca = pca.fit_transform(x.cpu().numpy())  # [H*W, 3]
        render_feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
        vis = (render_feature_vis - render_feature_vis.min()) / (render_feature_vis.max() - render_feature_vis.min())
    else:
        vis = (depth_normal + 1.) / 2.

    row0 = torch.cat([gt_image, image,], dim=2).cpu()
    row1 = torch.cat([depth_map, vis], dim=2).cpu()

    # image_to_show = torch.cat([row0, row1, row2], dim=1)
    image_to_show = torch.cat([row0, row1], dim=1)
    image_to_show = torch.clamp(image_to_show, 0, 1)
    
    os.makedirs(f"{out_path}/log_images/rgb", exist_ok = True)
    torchvision.utils.save_image(image_to_show, f"{out_path}/log_images/rgb/{iteration}.jpg")


def visualizer_semantic(render_pkg, iteration, out_path, attn_module, use_rgb=True, use_geo=False, use_ins=True):
    gt_image = render_pkg["gt_image"].cuda()
    image = render_pkg["render"].cuda() if render_pkg["render"] is not None else torch.zeros_like(gt_image).to(gt_image.device)
    pts = render_pkg["render_pts_world"].permute(1, 2, 0).cuda()

    instance_feature = render_pkg["render_ins_feature"].cuda()  # [D, H, W]
    instance_feature = instance_feature.permute(1, 2, 0) # From[D=16, H=730, W=988] to [H=730, W=988, D=16]

    H, W, D = instance_feature.shape
    x = instance_feature.reshape(-1, D)  # [H*W, D]
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x.cpu().numpy())  # [H*W, 3]
    render_feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
    render_feature_vis = (render_feature_vis - render_feature_vis.min()) / (render_feature_vis.max() - render_feature_vis.min())

    rgb = image
    if use_rgb:
        feature = attn_module.rgb_embed(rgb.permute(1, 2, 0).reshape(-1, 3)).reshape(H, W, -1)

    if use_ins:
        feature = torch.cat([feature, instance_feature], dim=-1)  # [H, W, C+D]

    if use_geo:
        geo_feature = attn_module.PEn(pts)
        feature = torch.cat([feature, geo_feature.reshape(H, W, -1)], dim=-1)

    D = feature.shape[-1]
    out_flat, _ = attn_module.inference(feature.reshape(-1, D).float(), pts.reshape(-1, 3).float())  # [H*W, D]

    recon_rgb = out_flat['rgb']  # [H*W, 3]
    recon_rgb = recon_rgb.reshape(H, W, 3).permute(2, 0, 1)
    recon_rgb = torch.clamp(recon_rgb, 0, 1)

    semantic_flat = out_flat['semantic']  # [H*W, semantic_D]
    x_pca = pca.fit_transform(semantic_flat.cpu().numpy())
    recon_semantic = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
    recon_semantic_vis = (recon_semantic - recon_semantic.min()) / (recon_semantic.max() - recon_semantic.min())
    
    if render_pkg["tgt_feature"] is not None:
        tgt_feature = render_pkg["tgt_feature"].cuda()
        D = tgt_feature.shape[0]
        tgt_flat = tgt_feature.permute(1, 2, 0).reshape(-1, D)  # [H*W, D]

        x_pca = pca.fit_transform(tgt_flat.cpu().numpy())
        tgt_feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
        tgt_feature_vis = (tgt_feature_vis - tgt_feature_vis.min()) / (tgt_feature_vis.max() - tgt_feature_vis.min())
    else:
        tgt_feature_vis = torch.zeros_like(recon_semantic_vis).to(recon_semantic_vis.device)
    
    row0 = torch.cat([gt_image, image, recon_rgb], dim=2).cpu()
    row1 = torch.cat([tgt_feature_vis, render_feature_vis, recon_semantic_vis], dim=2).cpu()

    image_to_show = torch.cat([row0, row1], dim=1)
    image_to_show = torch.clamp(image_to_show, 0, 1)
    
    os.makedirs(f"{out_path}/log_images/semantic", exist_ok = True)
    torchvision.utils.save_image(image_to_show, f"{out_path}/log_images/semantic/{iteration}.jpg")


def visualizer_slot(render_pkg, iteration, out_path, attn_module, use_rgb=False, use_geo=False, use_ins=True):
    gt_image = render_pkg["gt_image"].cuda()
    instance_feature = render_pkg["render_ins_feature"] .cuda() # [D, H, W]
    instance_feature = instance_feature.permute(1, 2, 0) # From[D=16, H=730, W=988] to [H=730, W=988, D=16]
    image = render_pkg["render"].cuda()
    pts = render_pkg["render_pts_world"].permute(1, 2, 0).cuda()

    H, W, D = instance_feature.shape
    rgb = image
    if use_rgb:
        feature = attn_module.rgb_embed(rgb.permute(1, 2, 0).reshape(-1, 3)).reshape(H, W, -1)

    if use_ins:
        feature = torch.cat([feature, instance_feature], dim=-1)  # [H, W, C+D]

    if use_geo:
        geo_feature = attn_module.PEn(pts)
        feature = torch.cat([feature, geo_feature.reshape(H, W, -1)], dim=-1)

    slots, _ = attn_module.get_slots()
    num_slots = slots.shape[0]
    os.makedirs(f"{out_path}/log_images/slot_visualization/{iteration}/", exist_ok = True)

    features, logits = attn_module.inference(feature.reshape(-1, feature.shape[-1]).float(), pts.reshape(-1, 3).float())  # [H*W, D]
    semantics = features['semantic']

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(semantics.cpu().numpy())  # [H*W, 3]
    feature_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1).to(gt_image.device)
    feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min())

    for i in range(num_slots):
        # attention heat map
        logit = logits[..., i].reshape(H, W)
        attn_map = apply_depth_colormap(logit[..., None], None, near_plane=0.0, far_plane=1.0).permute(2, 0, 1)
        attn_map_rgb = gt_image * logit[None]

        row0 = torch.cat([gt_image, feature_vis], dim=2).cpu()
        row1 = torch.cat([attn_map, attn_map_rgb], dim=2).cpu()

        # image_to_show = torch.cat([row0, row1, row2], dim=1)
        image_to_show = torch.cat([row0, row1], dim=1)
        image_to_show = torch.clamp(image_to_show, 0, 1)

        torchvision.utils.save_image(image_to_show, f"{out_path}/log_images/slot_visualization/{iteration}/slot_{i}.jpg")


def visualizer_ply(gaussians, iteration, out_path, attn_module, use_rgb=False, use_geo=False, use_ins=True, th=0.5):
    save_path = f"{out_path}/log_images/slot_ply/{iteration}/"
    os.makedirs(save_path, exist_ok = True)

    pts = gaussians.get_xyz()

    instance_feature = gaussians.get_ins_feature()

    shs = gaussians.get_features()
    rgb = SH2RGB(shs)

    if use_rgb:
        feature = attn_module.rgb_embed(rgb.reshape(-1, 3))

    if use_ins:
        feature = torch.cat([feature, instance_feature], dim=-1)  # [H, W, C+D]

    if use_geo:
        geo_feature = attn_module.PEn(pts)
        feature = torch.cat([feature, geo_feature], dim=-1)

    slots, _ = attn_module.get_slots()
    num_slots = slots.shape[0]
    features, logits = attn_module.inference(feature.float(), pts.float())  # [H*W, D]
    semantics = features['semantic']

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(semantics.cpu().numpy())  # [H*W, 3]
    feature_vis = torch.from_numpy(x_pca)
    feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min())
    feature_vis = torch.clamp(feature_vis, min=0.0, max=1.0)

    for i in range(num_slots):
        # attention heat map
        logit = logits[..., i]
        valid_mask = logit > th

        valid_pts = pts[valid_mask].cpu().numpy()  # [N, 3]
        valid_color = feature_vis[valid_mask].cpu().numpy()  # [N, 3]

        save_dir = os.path.join(save_path, f"slot_{i}.ply")

        if valid_pts.shape[0] == 0:
            continue

        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(valid_color.astype(np.float64))

            o3d.io.write_point_cloud(save_dir, pcd)
        else:
            save_ply(save_dir, valid_pts, valid_color)

    
