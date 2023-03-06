import os
from collections import namedtuple

import numpy as np
import torch

from ml.dqn.model import QNet
from ml.dqn.prioritized_memory import Memory

Experience = namedtuple("Experience", field_names=("s", "a", "r", "d", "s_"))


class DQN:
    def __init__(
        self,
        state_size,
        n_actions,
        device="cpu",
        lr=1e-3,
        alpha=0.99,
        reward_decay=0.99,
        target_replace_iter=1000,
        memory_size=1000000,
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
        self.optimizer = torch.optim.RMSprop(self.qnet.parameters(), lr=lr, alpha=alpha)
        self.criterion = torch.nn.MSELoss()

        self.memory = Memory(memory_size)

        self.learn_steps = 0

    def save_model(self, fname, iter):
        save_dir = os.path.join(os.path.dirname(__file__), "model")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            {"iter": iter, "QNet": self.qnet_target.state_dict()},
            os.path.join(save_dir, fname),
        )

    def load_model(self, fname):
        model_path = os.path.join(os.path.dirname(__file__), "model", fname)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ERROR: No model saved under the given path: {model_path}"
            )

        self.qnet.load_state_dict(
            torch.load(model_path, map_location=self.device)["QNet"]
        )

    def store_transition(self, s, a, r, d, s_):
        exp = Experience(s, a, r, d, s_)
        with torch.no_grad():
            s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            a = torch.LongTensor([a]).unsqueeze(0).to(self.device)
            r = torch.FloatTensor([r]).unsqueeze(0).to(self.device)
            d = torch.FloatTensor([int(d)]).unsqueeze(0).to(self.device)
            s_ = torch.FloatTensor(s_).unsqueeze(0).to(self.device)

            q_eval = self.qnet(s).gather(-1, a).squeeze(-1)
            q_next = self.qnet_target(s_).max(-1).values
            q_target = r + (1.0 - d) * self.reward_decay * q_next

            # TD error
            error = abs(q_eval - q_target)

        self.memory.add(error.item(), exp)

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
        mb, idxs, is_weights = self.memory.sample(self.batch_size)
        is_weights = torch.FloatTensor(np.array(is_weights)).to(self.device)

        mb_states, mb_actions, mb_rewards, mb_dones, mb_next_states = zip(*mb)
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

            # Update priority
            errors = abs(q_eval.detach() - q_target)
            for idx, error in zip(idxs, errors):
                self.memory.update(idx, error.item())

        loss = (is_weights * self.criterion(q_eval, q_target)).mean()

        # Updata parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        return loss.item()
