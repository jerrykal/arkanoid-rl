import torch.nn as nn


# Dueling DQN
class QNet(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
        )
        self.fc_value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, state):
        fc_out = self.fc(state)
        value = self.fc_value(fc_out)
        advantage = self.fc_advantage(fc_out)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
