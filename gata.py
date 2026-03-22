import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHead(nn.Module):
    """
    Geometry-aware pose branch.
    Predict 3D head pose parameters: yaw, pitch, roll.

    Input : [B, C, H, W]
    Output: [B, 3]
    """
    def __init__(self, in_channels=32, hidden_dim=64, out_dim=3):
        super(PoseHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)   # [B, C]
        pose = self.mlp(x)            # [B, 3] -> yaw, pitch, roll
        return pose


class AffineHead(nn.Module):
    """
    Map relative 3D pose difference (yaw, pitch, roll)
    to a 2x3 affine matrix.

    Input : [B, 3]
    Output: [B, 2, 3]
    """
    def __init__(self, in_dim=3, hidden_dim=32):
        super(AffineHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)
        )
        self._init_as_identity()

    def _init_as_identity(self):
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0
            ], dtype=torch.float))

    def forward(self, delta_pose):
        theta = self.mlp(delta_pose).view(-1, 2, 3)   # [B, 2, 3]
        return theta


class GeometryAwareAlignment(nn.Module):
    """
    Geometry-aware temporal alignment module.

    Inputs:
        feat_prev: [B, C, H, W]
        feat_curr: [B, C, H, W]

    Outputs:
        feat_prev_aligned: [B, C, H, W]
        pose_prev: [B, 3]
        pose_curr: [B, 3]
        theta: [B, 2, 3]
    """
    def __init__(self, in_channels=32, pose_hidden=64, affine_hidden=32):
        super(GeometryAwareAlignment, self).__init__()
        self.pose_head = PoseHead(
            in_channels=in_channels,
            hidden_dim=pose_hidden,
            out_dim=3
        )
        self.affine_head = AffineHead(
            in_dim=3,
            hidden_dim=affine_hidden
        )

    def warp_feature(self, feat, theta):
        """
        feat : [B, C, H, W]
        theta: [B, 2, 3]
        """
        B, C, H, W = feat.shape
        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        feat_warped = F.grid_sample(
            feat,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        return feat_warped

    def forward(self, feat_prev, feat_curr):
        # pose prediction
        pose_prev = self.pose_head(feat_prev)          # [B, 3]
        pose_curr = self.pose_head(feat_curr)          # [B, 3]

        # relative pose difference
        delta_pose = pose_curr - pose_prev             # [B, 3]

        # affine matrix
        theta = self.affine_head(delta_pose)           # [B, 2, 3]

        # warp previous feature to current frame
        feat_prev_aligned = self.warp_feature(feat_prev, theta)

        return feat_prev_aligned, pose_prev, pose_curr, theta