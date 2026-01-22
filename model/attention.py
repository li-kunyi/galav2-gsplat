import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
Codebook Structure Explanation:

Each codebook is a learnable matrix of shape [N, D], where:
    - N is the number of codebook entries (i.e., the number of codes or prototypes).
    - D is the dimensionality of each code vector (i.e., feature dimension).

There are two types of codebooks used in this model:

1. Instance Codebook:
   - Shape: [N, D_instance]
   - Each row represents a learnable embedding that encodes a reusable prototype of visual instance features.
   - These prototypes can represent common visual parts or instance patterns (e.g., chair legs, table edges).
   - The attention mechanism maps 2D instance features (queries) to these visual prototypes (keys).

2. Language Codebook:
   - Shape: [N, D_language]
   - Each row corresponds to a semantic embedding associated with the visual prototype from the instance codebook.
   - These embeddings are trained to align with language features (e.g., CLIP-based embeddings).
   - The output of attention is computed as a weighted combination over these vectors.

Column-wise:
   - Each column in the codebook corresponds to a latent feature dimension.
   - These dimensions do not have explicit semantic meanings, but are learned to be useful for modeling instance and semantic concepts.

In short:
    - Each row = one learned prototype.
    - Each column = one latent dimension of the feature space.
    - Instance codebook defines visual queries (keys), language codebook defines semantic values.
"""

class BinarizeIndicator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indicator): # indicator : (K,)
        # Get the subnetwork by sorting the scores and using the top k%
        return (indicator >= .5).float() # hard gate
    @staticmethod
    def backward(ctx, g):
        # Send the gradient g straight-through on the backward pass.
        return g # For STE 



class Codebook(nn.Module):
    def __init__(self, tensor=None, size=None, dim=None, learnable=True):
        super().__init__()
        self.learnable = learnable

        if tensor is not None:
            if learnable:
                self.entries = nn.Parameter(tensor.clone())
            else:
                self.register_buffer('entries', tensor.clone())
        else:
            assert size is not None and dim is not None, "Must provide tensor or (size, dim)"
            if learnable:
                self.entries = nn.Parameter(torch.randn(size, dim)) # row=size (codebook_size=128/64), col=dim (instance_dim=16)
            else:
                self.register_buffer('entries', torch.randn(size, dim))

    def forward(self):
        return self.entries

class AttentionModule(nn.Module):
    def __init__(self, query_dim=16, key_dim=16, value_dim=16, codebook_size=64,
                 instance_codebook=None, language_codebook=None, learnable_codebooks=True):
        """
        Initializes the attention module with the given parameters.
        Dimensions:
            -  N: codebook_size: 64
            -  Q: (H*W, 16)
            -  K: (N, 16)
            -  V: (N, 16)
            - output_projector: From (N, 16) to (N, 512)
            - output: (B, H, W, 512)
        """
        super().__init__()
        self.gate_function = False
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.norm_q = nn.LayerNorm(key_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(value_dim)
        self.output_projector = nn.Sequential(
            nn.Linear(value_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.scale = math.sqrt(key_dim) # Softmax scale factor: 4.0
        # Wrap passed tensors or initialize random codebooks
        if instance_codebook is not None:
            self.key_codebook = Codebook(tensor=instance_codebook, learnable=learnable_codebooks)            # (N, 16)
        else:
            self.key_codebook = Codebook(size=codebook_size, dim=key_dim, learnable=learnable_codebooks)     # (N, 16)

        if language_codebook is not None: 
            self.value_codebook = Codebook(tensor=language_codebook, learnable=learnable_codebooks)          # (N, 16)
        else:
            self.value_codebook = Codebook(size=codebook_size, dim=value_dim, learnable=learnable_codebooks) # (N, 16)

    def forward(self, query):
        """
        query: Tensor of shape (B, H, W, D)
        Output: (B, H, W, 512), attention weights: (B, H, W, N)
        """
        B, H, W, D = query.shape
        query_flat = query.view(B, -1, D)   # (B, H*W, D)
        Q = self.query_proj(query_flat)     # (B, H*W, key_dim=16)
        Q = self.norm_q(Q)                  # LayerNorm over Q

        K = self.key_codebook()             # (N, key_dim=16)
        K = self.norm_k(K)                  # LayerNorm over K

        V = self.value_codebook()           # (N, value_dim=16)
        V = self.norm_v(V)                  # LayerNorm over V

        attn_logits = torch.matmul(Q, K.T) / self.scale        # (B, H*W, N=64)
        attn_weights = F.softmax(attn_logits, dim=-1)          # (B, H*W, N=64)
        output = torch.matmul(attn_weights, V)                 # (B, H*W, key_dim=16)

        output = output + Q                                    # Residual connection
                
        output = output.view(B, H, W, -1)                      # (B, H, W, N=16)
        output = self.output_projector(output)                 # (B, H, W, 512)
        attn_weights = attn_weights.view(B, H, W, -1)          # (B, H, W, N=64)

        output = output / (output.norm(dim=-1, keepdim=True) + 1e-9)
        return output, attn_weights

    def active_code_fraction(self, attn_weights=None):
        """Return the % of codes currently ON."""
        total_codes = self.key_codebook.entries.shape[0]
        if self.gate_function:
            gate = (self.indicator >= 0.5).float()
            num_active = int(gate.sum().item())
            active_frac = num_active / total_codes
            return active_frac, num_active
        else:
            assert attn_weights is not None, "attn_weights must be used when gate_function=False"
            usage = attn_weights.argmax(dim=-1)  # (B, H, W)
            unique_codes = torch.unique(usage)
            num_active = unique_codes.numel()
            active_frac = num_active / total_codes
            return active_frac, num_active
            
    @classmethod
    def from_codebooks(cls, instance_codebook, language_codebook, learnable=True):
        return cls(
            query_dim=instance_codebook.shape[1],
            key_dim=instance_codebook.shape[1],
            value_dim=language_codebook.shape[1],
            codebook_size=instance_codebook.shape[0],
            instance_codebook=instance_codebook,
            language_codebook=language_codebook,
            learnable_codebooks=learnable
        )

    def save(self, path, attn_weights):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'attn_module': self.state_dict(),
            'codebook_size': self.key_codebook.entries.shape[0],
            'key_dim': self.key_codebook.entries.shape[1],
            'value_dim': self.value_codebook.entries.shape[1], 
        }, path + "/attn_module.pth")
        torch.save(attn_weights.cpu(), path + "/attn_weights.pt")
        torch.save({
            'instance_codebook': self.key_codebook.cpu(),
        }, path + "/instance_codebook.pth")
        torch.save({
            'language_codebook': self.value_codebook.cpu()
        }, path + "/language_codebook.pth")

    @classmethod
    def load(cls, path, query_dim, learnable_codebooks=True):
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            query_dim=query_dim,
            key_dim=checkpoint['key_dim'],
            value_dim=checkpoint['value_dim'],
            codebook_size=checkpoint['codebook_size'],
            learnable_codebooks=learnable_codebooks
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def compute_attention_entropy(self, attn_weights, path, visualize=False):
        # shape: (1, H, W, N) → (H, W, N)
        attn = attn_weights.squeeze(0)
        entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)  # (H, W)
        
        if visualize: 
            # Normalize the entropy map
            entropy_map = entropy.detach().cpu()
            entropy_map_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min())

            # Plot and save the figure
            plt.imshow(entropy_map_norm.cpu().numpy(), cmap='inferno')
            plt.title("Attention Entropy Map")
            plt.colorbar()

            # Save to file instead of showing
            plt.savefig(path + "/entropy_map.png")
            plt.close()  # Close the figure to free memory
            
        return entropy  # entropy map of shape (H, W)
    
    def sanity_check(self, attn_weights):
        # attn_weights: (1, H, W, N)
        attn_flat = attn_weights.reshape(-1, attn_weights.shape[-1])  # (H*W, N)
        usage = attn_flat.mean(dim=0)  # average attention per codebook entry

        # Plot and save the bar chart
        usage = usage.detach()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(usage)), usage.cpu().numpy())
        plt.title("Average Attention per Codebook Entry")
        plt.xlabel("Codebook Index")
        plt.ylabel("Mean Attention Weight")

        # Save to file instead of showing
        plt.savefig("sanity_check/codebook_usage.png")
        plt.close()  # Close to free memory
        top_entry = torch.argmax(usage)
        top_weight = usage[top_entry].item()
        print(f"[INFO] Most used codebook entry: {top_entry.item()} with avg weight {top_weight:.4f}")
        print(f"[INFO] Spread (max - min): {usage.max().item() - usage.min().item():.6f}")