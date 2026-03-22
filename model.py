import torch
import torch.nn as nn
from easydict import EasyDict as edict

from resnet import resnet18
from sam import DSDA
from mlp import STAGE_MLP
from gata import GeometryAwareAlignment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        maps = 32

       
        self.base_model = resnet18(pretrained=False, maps=maps)

       
        self.geo_align = GeometryAwareAlignment(
            in_channels=maps,
            pose_hidden=64,
            affine_hidden=32
        )

        
        self.sam = DSDA(input_dim=maps, dim=512)

       
        gru_config = edict({
            "camera_frame_type": "face",
            "face_net_rnn_num_features": 96,  # 3 * 32
            "tanh": True,
            "w_pog_loss": 1.0,
        })
        self.STAGE_MLP = STAGE_MLP(config=gru_config)

        self.gaze_loss_op = nn.L1Loss()
        self.pose_loss_op = nn.L1Loss()

       
        self.lambda_pose = 0.1

        self.to(device)

    def forward(self, x_t, return_aux=False):
        """
        x_t: [B, T, C, H, W], DGAGaze uses T=2
        """
        b, t, C, H, W = x_t.shape
        if t < 2:
            raise ValueError(f"DGAGaze requires at least 2 frames, but got t={t}")

       
        x_t_features = []
        for i in range(t):
            frame = x_t[:, i, :, :, :]
            feature_t = self.base_model(frame)   # [B, 32, 7, 7]
            x_t_features.append(feature_t)

        x_t_features = torch.stack(x_t_features, dim=1)  # [B, T, 32, 7, 7]

        sam_outputs = []
        pose_prev_preds = []
        pose_curr_preds = []

       
        for i in range(1, t):
            feat_prev = x_t_features[:, i - 1, :, :, :]  # [B, 32, 7, 7]
            feat_curr = x_t_features[:, i, :, :, :]      # [B, 32, 7, 7]

            
            feat_prev_aligned, pose_prev, pose_curr, theta = self.geo_align(feat_prev, feat_curr)

        # DSDA input must be aligned previous-frame feature + current-frame feature
            sam_output = self.sam(
                feat_prev_aligned.unsqueeze(1),   # [B, 1, 32, 7, 7]
                feat_curr.unsqueeze(1)            # [B, 1, 32, 7, 7]
            )
            sam_outputs.append(sam_output)

            pose_prev_preds.append(pose_prev)
            pose_curr_preds.append(pose_curr)

        sam_outputs = torch.stack(sam_outputs, dim=1)  # [B, T-1, 1, 96]
        sam_outputs_flat = sam_outputs.view(
            sam_outputs.size(0),
            sam_outputs.size(1),
            -1
        )  # [B, T-1, 96]

        mlp_input_dict = {'face_patch': sam_outputs_flat}
        mlp_output, _ = self.STAGE_MLP(mlp_input_dict)  # [B, T-1, 2]

        gaze = mlp_output[:, -1, :]  # final gaze of current frame

        aux = {
            "pose_prev_pred": pose_prev_preds[-1],
            "pose_curr_pred": pose_curr_preds[-1],
        }

        if return_aux:
            return gaze, aux
        return gaze

    def loss(self, x_t, gaze_label, pose_prev_label, pose_curr_label):
        gaze_pred, aux = self.forward(x_t, return_aux=True)

        pose_prev_pred = aux["pose_prev_pred"]
        pose_curr_pred = aux["pose_curr_pred"]

       
        gaze_loss = self.gaze_loss_op(gaze_pred, gaze_label)

        # auxiliary pose supervision using pseudo-labels
        pose_prev_loss = self.pose_loss_op(pose_prev_pred, pose_prev_label)
        pose_curr_loss = self.pose_loss_op(pose_curr_pred, pose_curr_label)
        pose_loss = 0.5 * (pose_prev_loss + pose_curr_loss)

        total_loss = gaze_loss + self.lambda_pose * pose_loss

        loss_dict = {
            "total_loss": total_loss.detach(),
            "gaze_loss": gaze_loss.detach(),
            "pose_loss": pose_loss.detach(),
            "pose_prev_loss": pose_prev_loss.detach(),
            "pose_curr_loss": pose_curr_loss.detach()
        }

        return total_loss, loss_dict
