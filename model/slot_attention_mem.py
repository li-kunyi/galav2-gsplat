import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

class Attention(nn.Module):
    def __init__(self, ins_dim, tgt_feat_dim, num_slots, in_slot_dim, tgt_slot_dim, iters=3, use_ins=True, use_rgb=True, use_geo=False):
        super().__init__()
        self.slot_iters = iters
        self.num_slots = num_slots
        self.avg_attn_mass = torch.zeros(num_slots, device='cuda:0')
        self.attn_count = 0
        self.attn_max = torch.zeros(num_slots, device='cuda:0')
        self.densify_count = torch.zeros(num_slots, device='cuda:0')

        if use_ins:
            in_feat_dim = ins_dim
        else:
            in_feat_dim = 0

        if use_geo:
            self.PEn = PositionalEncoding(learnable=True, out_dim=ins_dim)
            in_feat_dim += self.PEn.dim

        if use_rgb:
            self.rgb_embed = ColorEncoding(encode=True, out_dim=ins_dim)
            in_feat_dim += self.rgb_embed.dim
        
        # Initialize slots
        self.in_slots = torch.randn(num_slots, in_slot_dim, requires_grad=True, device='cuda:0')
        self.tgt_slots = torch.randn(num_slots, tgt_slot_dim, requires_grad=True, device='cuda:0')

        # Normalization and linear layers for features
        self.norm_input = nn.LayerNorm(in_feat_dim)
        self.linear_input = nn.Linear(in_feat_dim, in_slot_dim)  

        self.norm_tgt = nn.LayerNorm(tgt_feat_dim)
        self.linear_tgt = nn.Linear(tgt_feat_dim, tgt_slot_dim)

        # Normalization and linear layers for slots
        self.norm_in_slots = nn.LayerNorm(in_slot_dim)
        self.linear_in_slots = nn.Linear(in_slot_dim, in_slot_dim)

        self.norm_tgt_slots = nn.LayerNorm(tgt_slot_dim)
        self.linear_tgt_slots = nn.Linear(tgt_slot_dim, tgt_slot_dim)      

        # Residual linear layers
        self.linear_residual = nn.Linear(in_feat_dim, tgt_slot_dim)  

        # GRU cells for slot updates
        self.gru_in = nn.GRUCell(in_slot_dim, in_slot_dim)
        self.gru_tgt = nn.GRUCell(tgt_slot_dim, tgt_slot_dim)
        
        self.ln_semantic = nn.LayerNorm(tgt_slot_dim)
        self.ln_rgb_ins = nn.LayerNorm(in_slot_dim)

        self.mlp_rgb = nn.Sequential(
            nn.Linear(in_slot_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.mlp_ins = nn.Sequential(
            nn.Linear(in_slot_dim, 64),
            nn.ReLU(),
            nn.Linear(64, ins_dim)
        )

        self.mlp_semantic = nn.Sequential(
            nn.Linear(tgt_slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, tgt_feat_dim)
        )

        self.mlp_decoder = nn.Sequential(
            nn.Linear(in_slot_dim+3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 3 rgb + 1 mask
        )
                        
    def slot_attn(self, inputs, targets, in_slots, tgt_slots):
        # slots as queries
        query_input = self.linear_in_slots(self.norm_in_slots(in_slots))  # [N, D1]
        query_tgt = self.linear_tgt_slots(self.norm_tgt_slots(tgt_slots))  # [N, D2]

        # features as keys
        key_input = self.linear_input(self.norm_input(inputs))  # [M, D1]
        key_tgt = self.linear_tgt(self.norm_tgt(targets))  # [M, D2] 

        D1 = query_input.shape[-1]  # in_slot_dim
        D2 = query_tgt.shape[-1]  # tgt_slot_dim
        D = D1 + D2

        # Query, Key, Value
        # q = torch.cat([query_input, query_tgt], dim=-1)  # [N, D1 + D2]
        # k = torch.cat([key_input, key_tgt], dim=-1)  # [M, D1 + D2]
        # v = k

        q = query_input  # [N, D1]
        k = key_input  # [M, D1]
        v = torch.cat([key_input, key_tgt], dim=-1)  # [M, D1 + D2]

        # Attention
        logits = torch.matmul(q, k.T) / math.sqrt(D1)
        attn = F.softmax(logits, dim=-1)  # [N, M]
        updates = torch.matmul(attn, v)  # [N, D]

        updates_in = updates[:, :D1]
        updates_tgt = updates[:, D1:]

        # GRU update
        updated_in_slots = self.gru_in(updates_in, in_slots)
        updated_tgt_slots = self.gru_tgt(updates_tgt, tgt_slots)

        return updated_in_slots, updated_tgt_slots

    def decoder(self, coords, slots):
        B, K, D = slots.shape
        _, N, _ = coords.shape

        slots = slots.unsqueeze(2).expand(-1, -1, N, -1)   # [B, K, N, D]
        coords = coords.unsqueeze(1).expand(-1, K, -1, -1) # [B, K, N, 2]

        decoder_input = torch.cat([slots, coords], dim=-1)

        out = self.mlp_decoder(decoder_input)  # [B, K, N, 4]

        rgb = out[..., :3]
        mask_logits = out[..., 3:]

        masks = torch.softmax(mask_logits, dim=1)
        recon = torch.sum(masks * rgb, dim=1)  # [B, N, 3]

        return recon
    
    def cross_attn(self, inputs, in_slots, tgt_slots, pts):
        q = self.linear_input(self.norm_input(inputs))
        k = self.linear_in_slots(self.norm_in_slots(in_slots))
        v = self.linear_tgt_slots(self.norm_tgt_slots(tgt_slots))

        res = self.linear_residual(self.norm_input(inputs))

        M, D = k.shape

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        attn = F.softmax(logits, dim=-1)  # softmax over slots

        # Corss attention: semantic reconstruction
        out_semantic = torch.matmul(attn, v) #+ res
        semantic = self.mlp_semantic(self.ln_semantic(out_semantic)) 
        semantic = F.normalize(semantic)

        # Self attention: apperance reconstruction
        # out_rgb_ins = torch.matmul(attn, k) #+ q
        # out_rgbs_norm = self.ln_rgb_ins(out_rgb_ins)
        
        # rgb = self.mlp_rgb(out_rgbs_norm + q)
        # ins = self.mlp_ins(out_rgbs_norm)

        rgb = self.decoder(pts[None], in_slots[None]).squeeze(0)
        ins = rgb

        # Concatenate rgb and semantic outputs
        output = {}
        output['rgb'] = rgb
        output['ins'] = ins
        output['semantic'] = semantic

        return output, attn

    def forward(self, in_flat, tgt_flat, pts, momentum=0.995):
        # Slot Attention -> update slots
        in_slots_updates, tgt_slots_updates = self.slot_attn(in_flat, tgt_flat, self.in_slots, self.tgt_slots)

        # Update slots with EMA
        updated_in_slots = self.in_slots * momentum + in_slots_updates * (1 - momentum)
        updated_tgt_slots = self.tgt_slots * momentum + tgt_slots_updates * (1 - momentum)

        # Cross-Attention
        out_flat, attn = self.cross_attn(in_flat, updated_in_slots, updated_tgt_slots, pts)

        return out_flat, updated_in_slots, updated_tgt_slots, attn
    
    def inference(self, in_flat, pts, chunk_size=8192):
        N = in_flat.shape[0]

        out_list = {}
        out_list['rgb'] = []
        out_list['ins'] = []
        out_list['semantic'] = []
        logit_list = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = in_flat[start:end]  # [chunk, K]
            pts_chunk = pts[start:end]

            out_chunk, logit_chunk = self.cross_attn(chunk, self.in_slots, self.tgt_slots, pts_chunk)

            out_list['rgb'].append(out_chunk['rgb'])
            out_list['ins'].append(out_chunk['ins'])
            out_list['semantic'].append(out_chunk['semantic'])
            logit_list.append(logit_chunk)

        out_flat = {}
        out_flat['rgb'] = torch.cat(out_list['rgb'], dim=0).to(in_flat.device)
        out_flat['ins'] = torch.cat(out_list['ins'], dim=0).to(in_flat.device)
        out_flat['semantic'] = torch.cat(out_list['semantic'], dim=0).to(in_flat.device)
        logits = torch.cat(logit_list, dim=0).to(in_flat.device)

        return out_flat, logits

    def get_input_embedding(self, inputs):
        q = self.linear_input(self.norm_input(inputs))
        return q
    
    def get_slots(self):
        return self.in_slots, self.tgt_slots
    
    def update_slots(self, in_slots, tgt_slots):
        self.in_slots = in_slots.detach().requires_grad_(True)
        self.tgt_slots = tgt_slots.detach().requires_grad_(True)

    def add_attn_status(self, weights):
        # for pruning
        self.avg_attn_mass += weights.mean(dim=0)
        self.attn_count += 1

        weight_max = torch.max(weights, dim=0).values
        self.attn_max = torch.max(weight_max, self.attn_max)
            
    def densification_and_prune(self, mass_th=0.02, max_th=0.9, prune=True, densify=True, momentum=0.7):
        num_slots = self.in_slots.shape[0]
        avg_attn_mass = self.avg_attn_mass / self.attn_count
        print(f"Number of Slots, Before: {num_slots}")

        # Minimum slots: 16
        if num_slots >= 16 and prune:
            # Prune
            mass_valid_mask = (avg_attn_mass > mass_th)
            max_valid_mask = (self.attn_max > max_th)
            valid_mask = torch.logical_and(mass_valid_mask, max_valid_mask)

            if valid_mask.sum() < 8:
                _, valid_mask = torch.topk(avg_attn_mass, k=8, largest=True)
            
            self.in_slots = self.in_slots[valid_mask]
            self.tgt_slots = self.tgt_slots[valid_mask]
            self.densify_count = self.densify_count[valid_mask]

            avg_attn_mass = avg_attn_mass[valid_mask]

        # Maximum slots: 128
        num_slots = self.in_slots.shape[0]
        if num_slots >= 128 and densify:
            _, top_indices = torch.topk(avg_attn_mass, k=128, largest=True)
            
            self.in_slots = self.in_slots[top_indices]
            self.tgt_slots = self.tgt_slots[top_indices]

        elif num_slots < 128 and densify:
            # Densify
            _, top_indices = torch.topk(avg_attn_mass, k=6, largest=True)
            new_in_slots = self.in_slots[top_indices]
            new_tgt_slots = self.tgt_slots[top_indices]
            
            new_num, in_slot_dim = new_in_slots.shape
            new_num, tgt_slot_dim = new_tgt_slots.shape

            new_in_slots = momentum * new_in_slots + (1 - momentum) * torch.randn(new_num, in_slot_dim, requires_grad=True, device='cuda:0')
            new_tgt_slots = momentum * new_tgt_slots + (1 - momentum) * torch.randn(new_num, tgt_slot_dim, requires_grad=True, device='cuda:0')

            random_in_slots = torch.randn(2, in_slot_dim, requires_grad=True, device='cuda:0')
            random_tgt_slots = torch.randn(2, tgt_slot_dim, requires_grad=True, device='cuda:0')

            self.in_slots[top_indices] = momentum * self.in_slots[top_indices] + (1 - momentum) * torch.randn(new_num, in_slot_dim, requires_grad=True, device='cuda:0')
            self.tgt_slots[top_indices] = momentum * self.tgt_slots[top_indices] + (1 - momentum) * torch.randn(new_num, tgt_slot_dim, requires_grad=True, device='cuda:0')

            self.in_slots = torch.cat([self.in_slots, new_in_slots, random_in_slots], dim=0)
            self.tgt_slots = torch.cat([self.tgt_slots, new_tgt_slots, random_tgt_slots], dim=0)

        # Reset status
        self.num_slots = self.in_slots.shape[0]

        self.avg_attn_mass = torch.zeros(self.num_slots, device='cuda:0')
        self.attn_count = 0
        self.attn_max = torch.zeros(self.num_slots, device='cuda:0')
        self.densify_count = torch.zeros(self.num_slots, device='cuda:0')

        print(f"Number of Slots, After: {self.num_slots}")

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        ckpt = {
            "model_state": self.state_dict(),
            "in_slots": self.in_slots.detach().cpu(),
            "tgt_slots": self.tgt_slots.detach().cpu(),
        }

        torch.save(ckpt, os.path.join(path, "attn_module.pth"))

    def load(self, path, map_location="cpu", device="cuda:0"):
        ckpt = torch.load(
            os.path.join(path, "attn_module.pth"),
            map_location=map_location
        )

        self.load_state_dict(ckpt["model_state"], strict=True)

        self.in_slots = ckpt["in_slots"].to(device).detach().requires_grad_(True)
        self.tgt_slots = ckpt["tgt_slots"].to(device).detach().requires_grad_(True)
    
    def load_target_feature(self, target_feature_dir, image_name, H, W, encoder='clip'):
        target_feature_name = os.path.join(target_feature_dir, image_name.split('.')[0])
        
        masks = np.load(target_feature_name + '_seg_map.npy', allow_pickle=True).item()
        seg_map = torch.from_numpy(masks['l']).cuda()  # seg_map: torch.Size([H, W]), use level 'l'
        features = torch.from_numpy(np.load(target_feature_name + '_feats.npy', allow_pickle=True)).cuda().float() # feature_map: [N, D] or [N, h, w, D] (dinov3), use level 'l'

        seg_map = F.interpolate(seg_map.unsqueeze(0).unsqueeze(0).float(), 
                                size=(H, W), mode="nearest").squeeze(0).squeeze(0).long()

        if encoder == 'dinov3':
            feature_map, valid_mask = self.get_feature_map_dinov3(seg_map, features)
        else:
            feature_map, valid_mask = self.get_feature_map(seg_map, features)
       
        return feature_map, valid_mask
    
    @staticmethod
    def get_feature_map(seg_map, feature_map):
        H, W = seg_map.shape

        y, x = torch.meshgrid(torch.arange(0, H, device='cuda'), torch.arange(0, W, device='cuda'))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        seg = seg_map[y, x].squeeze(-1).long()
        mask = seg != -1
        _point_feature = feature_map[seg].squeeze(0)
        mask = mask.reshape(H, W)
        
        point_feature = _point_feature.reshape(H, W, -1).permute(2, 0, 1)
       
        return point_feature, mask
    
    @staticmethod
    def get_feature_map_dinov3(seg_map, patch_feats):
        """
        seg_map: (H, W), segment id for each pixel, -1 indicates ignore
        patch_feats: (1, N_patches, D), patch-level feature map from DINO
        returns:
            dense_feature: (D, H, W)
            mask: (H, W), True for valid pixels
        """

        H, W = seg_map.shape
        seg_ids = seg_map.unique()
        seg_ids = seg_ids[seg_ids != -1]  # Ignore -1 values

        D = patch_feats.shape[-1]  # Feature dimension, e.g., 1280

        # Initialize dense feature map
        dense_feature = torch.zeros(D, H, W, device=patch_feats.device)
        mask = seg_map != -1

        for seg_id in seg_ids:
            # Current segment mask
            seg_mask = seg_map == seg_id  # (H, W), bool

            if seg_mask.sum() == 0:
                continue

            # Bounding box of the segment
            coords = seg_mask.nonzero(as_tuple=False)  # (N_pixels, 2)
            y1, x1 = coords.min(0)[0]
            y2, x2 = coords.max(0)[0] + 1

            h = y2-y1
            w = x2-x1
            long_side = max(w, h)

            cx = long_side // 2
            cy = long_side // 2
            _x1 = cx - w // 2
            _y1 = cy - h // 2
            _x2 = _x1 + w
            _y2 = _y1 + h

            cropped = seg_mask[y1:y2, x1:x2]
            seg_mask_square = torch.zeros(long_side, long_side, dtype=torch.bool).to(cropped.device)
            seg_mask_square[_y1:_y2, _x1:_x2] = cropped

            # Patch-level feature map: (D, H_patch, W_patch)
            patch_map = patch_feats[seg_id].permute(2, 0, 1)  # square size

            # Upsample to bounding box size
            seg_feats_square = F.interpolate(patch_map.unsqueeze(0), size=(long_side, long_side),
                                    mode='bilinear', align_corners=False).squeeze(0)  # (D, h_box, w_box)

            # Only write back to pixels belonging to the current segment
            dense_feature[:, seg_mask] = seg_feats_square[:, seg_mask_square]

        return dense_feature, mask

    
    @staticmethod
    def load_gt_image(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")

        transform = T.ToTensor()
        gt_image = transform(img).cuda()  # [C, H, W]，float32

        return gt_image


class PositionalEncoding(nn.Module):
    """
    Fourier Feature Positional Encoding for 3D points.
    x: tensor of shape (..., 3)
    L: number of frequency bands
    """
    def __init__(self, num_frequencies=10, include_xyz=True, learnable=False, out_dim=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_xyz = include_xyz
        self.learnable = learnable

        if self.learnable:
            self.linear = nn.Linear(3, out_dim)
        
            self.dim = out_dim
        else:
            self.dim = 3 * 2 * num_frequencies + 3 if self.include_xyz else 3 * 2 * num_frequencies
            # [2^0, 2^1, ..., 2^(L-1)]
            self.freq_bands = 2.0 ** torch.arange(num_frequencies)

    def forward(self, x):
        """
        x: (..., 3) 3D coordinates
        returns: (..., 3*2*num_frequencies)
        """
        if self.learnable:
            C = x.shape[-1]

            x = x.view(-1, C)    # [H*W, C]
            out = self.linear(x)           # [H*W, D]
        else:
            out = [x] if self.include_xyz else []
            for freq in self.freq_bands:
                out.append(torch.sin(freq * x))
                out.append(torch.cos(freq * x))
            out = torch.cat(out, dim=-1)

        return out

class ColorEncoding(nn.Module):
    """
    Fourier Feature Positional Encoding for 3D points.
    x: tensor of shape (..., 3)
    L: number of frequency bands
    """
    def __init__(self, encode=True, out_dim=16):
        super().__init__()
        self.encode = encode

        if self.encode:
            self.linear = nn.Linear(3, out_dim)
            self.dim = out_dim
        else:
            self.dim = 3
        
    def forward(self, x):
        if self.encode:
            C = x.shape[-1]

            x = x.view(-1, C)    # [H*W, C]
            out = self.linear(x)           # [H*W, D]
        else:
            out = x
        return out