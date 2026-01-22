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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def constrastive_clustering_loss(instance_features, gt_instance_masks):
    # hyperparameter, todo
    min_pixnum = 20
    
    valid_mask_idx = gt_instance_masks >= 0
    gt_valid_mask = gt_instance_masks[valid_mask_idx].long()
    ins_feat = instance_features[valid_mask_idx.squeeze(), :]
    
    ins_feat = ins_feat / (torch.norm(ins_feat, dim=-1, keepdim=True) + 1e-6).detach()
    
    # constrastive clustering
    cluster_ids, cnums_all = torch.unique(gt_valid_mask, return_counts=True)
    cluster_ids = cluster_ids[cnums_all > min_pixnum]  # filter
    cnums = cnums_all[cnums_all > min_pixnum]
    cnum = cluster_ids.shape[0] # cluster number
    
    # mean feature
    cluster_masks = (gt_valid_mask.unsqueeze(0) == cluster_ids.unsqueeze(1)).float()
    sum_features = torch.matmul(ins_feat.t(), cluster_masks.t()) 
    u_list = sum_features / cnums.unsqueeze(0).float()
    
    # temperature
    expanded_u_list = torch.matmul(u_list, cluster_masks)
    norms = torch.norm(ins_feat.t() - expanded_u_list, dim=0, keepdim=True)
    masked_norms = norms * cluster_masks
    phi_list = masked_norms.sum(dim=1) / (cnums * torch.log(cnums + 10) + 1e-6)
    #phi_list = phi_list.squeeze()
    u_list = u_list.t()
    
    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
    phi_list = phi_list.detach()
    
    loss = 0.0
    for i in range(cnum):
        
        cluster_mask = cluster_masks[i]
        cluster_features = ins_feat[cluster_mask.bool(), :]
        centroid = u_list[i, :].unsqueeze(1)
        
        positive_sim = torch.exp(torch.sum(cluster_features.T * centroid, dim=0) / phi_list[i])
        all_sims = torch.exp(torch.matmul(cluster_features, u_list.T) / phi_list.T)
        
        log = -torch.log(positive_sim / (all_sims[:, :].sum(dim=1, keepdim=True) + 1e-6))
        cluster_loss = torch.sum(log.mean(dim=1))
        loss += cluster_loss
        
    loss /= len(u_list)
    return loss

def cosine_similarity(predicted, target, reduction='mean'):    
    """
    Computes the mean cosine distance loss between predicted and target tensors.
        :param predicted (Tensor): Predicted features of shape (batch_size, feature_dim).
        :param target (Tensor): Target features of the same shape as predicted.
    Returns: Tensor: Scalar loss value (mean cosine distance).
    """
    # predicted_normalized = F.normalize(predicted.reshape(-1, 512), p=2, dim=-1)
    # target_normalized = F.normalize(target.reshape(-1, 512), p=2, dim=-1)

    # cosine_sim = torch.sum(predicted_normalized.reshape(-1, 512) * target_normalized.reshape(-1, 512), dim=-1)
    cosine_sim = F.cosine_similarity(predicted.reshape(-1, 512), target.reshape(-1, 512), dim=-1)
    cosine_distance = 1 - cosine_sim
    
    # Calculate the loss
    if reduction == 'mean':
        loss = cosine_distance.mean()
    elif reduction == 'sum':
        loss = cosine_distance.sum()
    else:
        loss = cosine_distance  # No reduction
    
    # segment_losses = {}
    # predicted = predicted.reshape(-1, 512)
    # target = target.reshape(-1, 512)
    # unique_segments = torch.unique(level_seg)
    # for seg_id in unique_segments:
    #     mask = (level_seg == seg_id).squeeze()
    #     if mask.sum() > 0:
    #         seg_pred = predicted[mask]
    #         seg_target = target[mask]
            
    #         seg_loss = (1 - torch.sum(seg_pred * seg_target, dim=-1)).mean()
    #         segment_losses[seg_id.item()] = seg_loss
            
    # loss = sum(segment_losses.values()) / len(segment_losses)
    
    return loss

def entropy_loss(attn_weights: torch.Tensor, eps: float = 1e-8, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the entropy of a probability distribution and return a scalar loss (to be minimized).
        :param attn_weights (Tensor): attention weights, shape (B, H, W, N), where sum over the last dim is 1.
        :param eps (float): small constant for numerical stability in log.
        :param reduction (str): 'none' | 'mean' | 'sum'
            - 'none': returns entropy per probability vector, shape (...)
            - 'mean': returns averaged scalar
            - 'sum' : returns summed scalar
    
    Returns: Tensor: scalar (if reduction is 'mean' or 'sum') or Tensor of shape (...) for 'none'.
    """
    # p * log(p)
    ent = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1) # Shape: (B, H, W)
    
    if reduction == 'none':
        return ent
    elif reduction == 'sum':
        return ent.sum()
    elif reduction == 'mean':
        return ent.mean()
    else:
        raise ValueError(f"reduction must be one of ['none','mean','sum'], got {reduction}")

        