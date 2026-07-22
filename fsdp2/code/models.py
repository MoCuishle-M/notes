import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyViT(nn.Module):
    """仅用于抽取图像特征，极度压缩尺寸"""
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 embed_dim=128, depth=2, num_heads=4):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, use_moe=False)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, use_moe=False, num_experts=2, top_k=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if use_moe:
            self.ffn = MoEFFN(dim, dim * 4, num_experts, top_k)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MoEFFN(nn.Module):
    """简单的 Mixture of Experts，每个专家是一个 FFN"""
    def __init__(self, dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim) * 0.02)
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[..., k]
            w = weights[..., k:k+1]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    selected = x[mask]
                    h = selected @ self.w1[e]
                    h = F.gelu(h)
                    expert_out = h @ self.w2[e]
                    out[mask] += w[mask] * expert_out
        return out


class TinyLLM(nn.Module):
    def __init__(self, vocab_size=1000, dim=128, depth=2, num_heads=4,
                 num_experts=2, top_k=1, max_seq_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, use_moe=True,
                             num_experts=num_experts, top_k=top_k)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, vision_features=None):
        B, T = input_ids.shape
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :T, :]
        if vision_features is not None:
            x = torch.cat([vision_features, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class MultiModalModel(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 vit_embed_dim=128, vit_depth=2, vit_heads=4,
                 vocab_size=1000, llm_dim=128, llm_depth=2, llm_heads=4,
                 num_experts=2, top_k=1, max_seq_len=128):
        super().__init__()
        self.vit = TinyViT(image_size, patch_size, in_channels,
                           vit_embed_dim, vit_depth, vit_heads)
        self.llm = TinyLLM(vocab_size, llm_dim, llm_depth, llm_heads,
                           num_experts, top_k, max_seq_len)
        self.vision_proj = nn.Linear(vit_embed_dim, llm_dim) if vit_embed_dim != llm_dim else nn.Identity()

    def forward(self, images, input_ids, labels=None):
        vis = self.vit(images)
        vis = self.vision_proj(vis)
        logits = self.llm(input_ids, vis)
        if labels is not None:
            loss = F.cross_entropy(logits[:, vis.size(1):].reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
            return logits, loss
        return logits
