import torch
import torch.nn as nn
import numpy as np

half_pi = 0.5 * np.pi

class STAGE_MLP(nn.Module):
    def __init__(self, config):
        super(STAGE_MLP, self).__init__()
        self.config = config
        self.input_dim = config.face_net_rnn_num_features
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 2),  # Output 2D gaze coordinates
            nn.Tanh()
        )
        
        # Initialize weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_dict):
        output_dict = {}
        sam_features = input_dict['face_patch']  # [b, t-1, feature_dim]
        batch_size, t, _ = sam_features.size()

        if sam_features.dim() > 3:
            sam_features = sam_features.view(batch_size, t, -1)  # Flatten to [b, t, feature_dim]

        expected_dim = self.config.face_net_rnn_num_features
        if sam_features.size(-1) != expected_dim:
            raise ValueError(f"Expected SAM feature dimension {expected_dim}, but got {sam_features.size(-1)}")

        # MLP processing
        mlp_out = self.mlp(sam_features)  # [b, t-1, 2]
        output_dict['pred'] = half_pi * mlp_out if self.config.tanh else mlp_out
        return output_dict['pred'], None  # Return None for hidden state to match GRU interface