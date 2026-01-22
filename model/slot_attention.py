import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlotAttention(nn.Module):
    def __init__(self, in_feat_dim, tgt_feat_dim, num_slots, in_slot_dim, tgt_slot_dim, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.in_slot_dim = in_slot_dim
        self.tgt_slot_dim = tgt_slot_dim
        self.iters = iters

        # Slot update
        self.gru_in = nn.GRUCell(in_slot_dim, in_slot_dim)
        self.mlp_in = nn.Sequential(
            nn.Linear(in_slot_dim, in_slot_dim),
            nn.ReLU(),
            nn.Linear(in_slot_dim, in_slot_dim)
        )

        self.gru_tgt = nn.GRUCell(tgt_slot_dim, tgt_slot_dim)
        self.mlp_tgt = nn.Sequential(
            nn.Linear(tgt_slot_dim, tgt_slot_dim),
            nn.ReLU(),
            nn.Linear(tgt_slot_dim, tgt_slot_dim)
        )

        self.norm_in_slots = nn.LayerNorm(in_slot_dim)
        self.norm_tgt_slots = nn.LayerNorm(tgt_slot_dim)

        self.norm_input = nn.LayerNorm(in_feat_dim)
        self.norm_target = nn.LayerNorm(tgt_feat_dim)

        # Linear maps for attention
        self.to_q_in = nn.Linear(in_slot_dim, in_slot_dim)
        self.to_k_in = nn.Linear(in_feat_dim, in_slot_dim)  
        self.to_v_in = nn.Linear(in_feat_dim, in_slot_dim)

        self.to_v_tgt = nn.Linear(tgt_feat_dim, tgt_slot_dim)

    def forward(self, inputs, target, in_slots, tgt_slots):
        # Precompute keys and values
        inputs_norm = self.norm_input(inputs)  # [N, C]
        in_k = self.to_k_in(inputs_norm)  # [M, D]
        in_v = self.to_v_in(inputs_norm)  # [M, D]
        M, D = in_k.shape

        target_norm = self.norm_target(target)
        tgt_v = self.to_v_tgt(target_norm)  # [M, D]

        # Iterative slot update
        for _ in range(self.iters):
            in_slots_norm = self.norm_in_slots(in_slots)
            in_q = self.to_q_in(in_slots_norm)  # [N, D]

            # Attention logits [N, M]
            logits = torch.matmul(in_q, in_k.T) / math.sqrt(D)
            attn = F.softmax(logits, dim=0)  # softmax over slots

            # Aggregate features
            in_updates = torch.matmul(attn, in_v)  # [N, D]
            tgt_updates = torch.matmul(attn, tgt_v)

            # GRU update
            in_slots = self.gru_in(in_updates, in_slots)
            tgt_slots = self.gru_tgt(tgt_updates, tgt_slots)
            # MLP residual
            in_slots = in_slots + self.mlp_in(in_slots)
            tgt_slots = tgt_slots + self.mlp_tgt(tgt_slots)

        return in_slots, tgt_slots
    
    
class CrossAttention(nn.Module):
    def __init__(self, in_feat_dim, tgt_feat_dim, in_slot_dim, tgt_slot_dim):
        super().__init__()
        self.norm_input = nn.LayerNorm(in_feat_dim)
        self.norm_in_slots = nn.LayerNorm(in_slot_dim)
        self.norm_tgt_slots = nn.LayerNorm(tgt_slot_dim)

        self.to_q = nn.Linear(in_feat_dim, in_slot_dim)  
        self.to_k = nn.Linear(in_slot_dim, in_slot_dim)
        self.to_v = nn.Linear(tgt_slot_dim, tgt_slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(tgt_slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, tgt_feat_dim)
        )

    def forward(self, inputs, in_slots, tgt_slots):

        # Prepare keys and values from target features
        inputs_norm = self.norm_input(inputs)
        q = self.to_q(inputs_norm)  # [M, D]

        in_slots_norm = self.norm_in_slots(in_slots)
        k = self.to_k(in_slots_norm)  # [N, D]

        M, D = k.shape

        tgt_slots_norm = self.norm_tgt_slots(tgt_slots)
        v = self.to_v(tgt_slots_norm)  # [N, D]

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        attn = F.softmax(logits, dim=1)  # softmax over target features

        # Aggregate features
        out_flat = torch.matmul(attn, v)  # [N, D]

        output = self.mlp(out_flat)  

        return output, attn


class Attention(nn.Module):
    def __init__(self, in_feat_dim, tgt_feat_dim, num_slots, in_slot_dim, tgt_slot_dim, iters=3, train=True):
        super().__init__()
        self.in_slots = torch.randn(num_slots, in_slot_dim, requires_grad=True)
        self.tgt_slots = torch.randn(num_slots, tgt_slot_dim, requires_grad=True)

        if train:
            self.slot_attn = SlotAttention(in_feat_dim, tgt_feat_dim, num_slots, in_slot_dim, tgt_slot_dim, iters)
        self.cross_attn = CrossAttention(in_feat_dim, tgt_feat_dim, in_slot_dim, tgt_slot_dim)

    def forward(self, in_flat, tgt_flat):
        # Slot Attention -> update slots, get shared attn
        updated_in_slots, updated_tgt_slots = self.slot_attn(in_flat, tgt_flat, self.in_slots, self.tgt_slots)

        # Cross-Attention
        out_flat, logits = self.cross_attn(in_flat, updated_in_slots, updated_tgt_slots)

        return out_flat, updated_in_slots, updated_tgt_slots, logits
    
    def inference(self, in_flat):
        out_flat, logits = self.cross_attn(in_flat, self.in_slots, self.tgt_slots)
        return out_flat
    
    def update_slots(self, in_slots, tgt_slots):
        self.in_slots = in_slots.detach().require_grad_(True)
        self.tgt_slots = tgt_slots.detach().require_grad_(True)

    def densification_and_prune(self, feats, th=0.7):
        with torch.no_grad():
            pass