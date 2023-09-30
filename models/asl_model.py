import torch
import torch.nn as nn


class WTALModel(nn.Module):
    def __init__(self, config):
        super(WTALModel, self).__init__()
        self.len_feature = config.len_feature
        self.num_classes = config.num_classes

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        input = x.permute(0, 2, 1)

        cas_fg = self.base_module(input).permute(0, 2, 1)
        action_flow = torch.sigmoid(self.action_module_flow(input[:, 1024:, :]))
        action_rgb = torch.sigmoid(self.action_module_rgb(input[:, :1024, :]))

        return cas_fg, action_flow, action_rgb
