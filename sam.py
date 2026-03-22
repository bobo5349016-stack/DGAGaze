import torch
from torch import nn
from torch.nn.init import xavier_uniform_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, alpha=0.1):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.fc_diff1 = nn.Linear(channels, channels // reduction)
        self.fc_diff2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha

    def forward(self, x, input_diff=None):
        b, c, h, w = x.size()
        y = x.mean((2, 3))  # (b, c)
        y = self.fc1(y)
        y = self.relu(y)
        w = self.sigmoid(self.fc2(y))
        if input_diff is not None:
            diff = input_diff.mean((2, 3))  # (b, c)
            diff = self.fc_diff1(diff)
            diff = self.relu(diff)
            w_diff = self.sigmoid(self.fc_diff2(diff))
            w = w * (1 + self.alpha * w_diff)
        w = w.view(b, c, 1, 1).expand_as(x)
        return x * w
def simam_attention(x, e_lambda=1e-4):
    b, c, h, w = x.size()
    n = h * w - 1

    
    mean = x.mean(dim=(2,3), keepdim=True)
    
    var = ((x - mean)**2).sum(dim=(2,3), keepdim=True) / n
    
    e = (x - mean)**2 / (4*(var + e_lambda)) + 0.5
    att = torch.sigmoid(-e)
    return x * att
    
class DSDA(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim

        self.embed = nn.Sequential(
            nn.Conv2d(self.input_dim * 2, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.embed_diff = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.se = SEBlock(self.dim, reduction=16, alpha=0.1)
        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)

    def forward(self, input_1, input_2):
        batch_size, t, K1, H, W = input_1.size()
        input_diff = input_2 - input_1

        input_before = torch.cat([input_1, input_diff], 2)
        input_after = torch.cat([input_2, input_diff], 2)

        input_before = input_before.view(batch_size * t, 2 * K1, H, W)
        input_after = input_after.view(batch_size * t, 2 * K1, H, W)
        input_diff_reshaped = input_diff.view(batch_size * t, K1, H, W)

        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        embed_diff = self.embed_diff(input_diff_reshaped)

        embed_before = self.se(embed_before, embed_diff)
        embed_after = self.se(embed_after, embed_diff)

        
        attended_before = simam_attention(embed_before)
        attended_after  = simam_attention(embed_after)

        
        input_1_reshaped = input_1.view(batch_size * t, K1, H, W)
        input_2_reshaped = input_2.view(batch_size * t, K1, H, W)
        att_weight_before = attended_before.mean(dim=1, keepdim=True)  # (batch_size * t, 1, H, W)
        att_weight_after = attended_after.mean(dim=1, keepdim=True)   # (batch_size * t, 1, H, W)
        att_1_expand = att_weight_before.repeat(1, K1, 1, 1)  # (batch_size * t, K1, H, W)
        att_2_expand = att_weight_after.repeat(1, K1, 1, 1)   # (batch_size * t, K1, H, W)
        attended_1 = (input_1_reshaped * att_1_expand).sum(2).sum(2)  # (batch_size * t, K1)
        attended_2 = (input_2_reshaped * att_2_expand).sum(2).sum(2)  # (batch_size * t, K1)

        attended_1 = attended_1.view(batch_size, t, -1)  # (batch_size, t, K1)
        attended_2 = attended_2.view(batch_size, t, -1)  # (batch_size, t, K1)
        input_attended = attended_2 - attended_1         # (batch_size, t, K1)

        output = torch.cat((attended_1, input_attended, attended_2), dim=-1)  # (batch_size, t, 3 * K1)

        return output