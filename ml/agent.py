import os
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from ml.model import QNet

Experience = namedtuple("Experience", field_names=("s", "a", "r", "d", "s_"))


class DQN:
    def __init__(
        self,
        state_size,
        n_actions,
        device="cpu",
        lr=1e-3,
        reward_decay=0.99,
        target_replace_iter=1000,
        memory_size=10000,
        batch_size=64,
    ):
        self.state_size = state_size
        self.n_actions = n_actions
        self.device = device
        self.reward_decay = reward_decay
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size

        self.qnet = QNet(state_size, n_actions).to(device)
        self.qnet_target = QNet(state_size, n_actions).to(device)
        self.qnet_target.eval()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        self.memory = deque(maxlen=memory_size)

        self.learn_steps = 0

    def save_model(self, fname, iter):
        save_dir = "./ml/model"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            {"iter": iter, "QNet": self.qnet_target.state_dict()},
            os.path.join(save_dir, fname),
        )

    def load_model(self, fname):
        model_path = f"./ml/model/{fname}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ERROR: No model saved under the given path: {model_path}")

        self.qnet.load_state_dict(torch.load(model_path)["QNet"])

    def store_transition(self, s, a, r, d, s_):
        self.memory.append(Experience(s, a, r, d, s_))

    def choose_action(self, state, epsilon=0):
        if np.random.random() <= epsilon:
            return np.random.randint(self.n_actions)
        else:
            state_tensor = torch.FloatTensor(state[np.newaxis, ...]).to(self.device)
            return self.qnet(state_tensor).argmax(dim=-1).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        # Replace target qnet with main qnet every self.target_replace_iter steps
        if self.learn_steps % self.target_replace_iter == 0:
            self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Sample minibatch of transition datas from memory
        sample_indices = np.random.choice(
            len(self.memory), self.batch_size, replace=False
        )
        mb_states, mb_actions, mb_rewards, mb_dones, mb_next_states = zip(
            *[self.memory[i] for i in sample_indices]
        )
        mb_states = torch.FloatTensor(np.array(mb_states)).to(self.device)
        mb_actions = torch.LongTensor(np.array(mb_actions)).to(self.device)
        mb_rewards = torch.FloatTensor(np.array(mb_rewards)).to(self.device)
        mb_dones = torch.FloatTensor(np.array(mb_dones)).to(self.device)
        mb_next_states = torch.FloatTensor(np.array(mb_next_states)).to(self.device)

        # Calculate loss
        q_eval = self.qnet(mb_states).gather(-1, mb_actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next = self.qnet_target(mb_next_states).max(-1).values
        q_target = mb_rewards + (1.0 - mb_dones) * self.reward_decay * q_next
        loss = F.mse_loss(q_eval, q_target)

        # Updata parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        return loss.item()
