import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class LayerNorm(nn.Module):


    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class MultiHeadSelfAttention(nn.Module):


    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):


    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):


    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NOCS_Predictor(nn.Module):


    def __init__(self, cfg):
        super().__init__()
        self.bins_num = cfg.bins_num
        self.cat_num = cfg.cat_num


        bin_lenth = 1 / self.bins_num
        half_bin_lenth = bin_lenth / 2
        self.register_buffer(
            'bins_center',
            torch.linspace(-0.5 + half_bin_lenth, 0.5 - half_bin_lenth, self.bins_num).view(-1, 1)
        )


        self.nocs_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 512, 1),
            LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 256, 1),
            LayerNorm(256),
        )

        self.residual_proj = nn.Conv1d(256, 256, 1)


        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(256, num_heads=8, dropout=0.1)
            for _ in range(2)  
        ])


        self.atten_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 512, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(512, self.cat_num * 3 * self.bins_num, 1),
        )

    def forward(self, kpt_feature, index):

        b, kpt_num, c = kpt_feature.shape


        x = kpt_feature.transpose(1, 2)
        residual = self.residual_proj(x)
        x = self.nocs_mlp(x) + residual
        x = x.transpose(1, 2)


        for block in self.transformer_blocks:
            x = block(x)


        attn = self.atten_mlp(x.transpose(1, 2))
        attn = attn.view(b * self.cat_num, 3 * self.bins_num, kpt_num).contiguous()
        attn = torch.index_select(attn, 0, index)
        attn = attn.view(b, 3, self.bins_num, kpt_num).contiguous()
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 3, 1, 2)

        # 计算NOCS坐标
        kpt_nocs = torch.matmul(attn, self.bins_center).squeeze(-1)
        return kpt_nocs


class PoseSizeEstimator(nn.Module):


    def __init__(self, dropout=0.1):
        super().__init__()

 
        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
        )

        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
        )


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 256, 512, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )

    
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )


        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 6),
        )


        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )


        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, pts1, pts2, pts1_local):

        feat1 = self.pts_mlp1(pts1.transpose(1, 2))  # (b, 64, n)
        feat2 = self.pts_mlp2(pts2.transpose(1, 2))  # (b, 64, n)


        feat1_t = feat1.transpose(1, 2)  # (b, n, 64)
        feat2_t = feat2.transpose(1, 2)  # (b, n, 64)
        feat1_attn, _ = self.cross_attn(feat1_t, feat2_t, feat2_t)
        feat1 = feat1_attn.transpose(1, 2)  # (b, 64, n)


        pose_feat = torch.cat([
            feat1,
            pts1_local.transpose(1, 2),
            feat2
        ], dim=1)

        pose_feat = self.pose_mlp1(pose_feat)


        pose_global_max = F.adaptive_max_pool1d(pose_feat, 1)
        pose_global_avg = F.adaptive_avg_pool1d(pose_feat, 1)
        pose_global = pose_global_max + pose_global_avg


        pose_feat = torch.cat([
            pose_feat,
            pose_global.expand_as(pose_feat)
        ], dim=1)

        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)


        r = self.rotation_estimator(pose_feat)
        r = self.ortho6d_to_rotation_matrix(
            r[:, :3].contiguous(),
            r[:, 3:].contiguous()
        )

        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)

        return r, t, s

    @staticmethod
    def ortho6d_to_rotation_matrix(x, y):
        """6D rotation representation to rotation matrix"""
        b = x.shape[0]
        x = F.normalize(x, dim=1)
        z = torch.cross(x, y, dim=1)
        z = F.normalize(z, dim=1)
        y = torch.cross(z, x, dim=1)
        return torch.stack([x, y, z], dim=2)



if __name__ == "__main__":

    class Config:
        bins_num = 32
        cat_num = 6

        class AttnLayer:
            pass


    cfg = Config()


    nocs_model = NOCS_Predictor(cfg)
    kpt_feature = torch.randn(2, 128, 256)
    index = torch.tensor([0, 1])
    nocs_output = nocs_model(kpt_feature, index)
    print(f"NOCS output shape: {nocs_output.shape}")


    pose_model = PoseSizeEstimator()
    pts1 = torch.randn(2, 1024, 3)
    pts2 = torch.randn(2, 1024, 3)
    pts1_local = torch.randn(2, 1024, 256)
    r, t, s = pose_model(pts1, pts2, pts1_local)
    print(f"Rotation shape: {r.shape}, Translation shape: {t.shape}, Size shape: {s.shape}")