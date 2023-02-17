import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, state):
        return self.fc(state)
