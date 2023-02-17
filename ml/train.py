import time

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from ml.agent import DQN


def get_epsilon(total_steps, epsilon_max=1.0, epsilon_min=0.02, epsilon_decay=500000):
    return max(epsilon_min, epsilon_max - total_steps / epsilon_decay)


class MLPlay:
    def __init__(self, *args, **kwargs):
        self.episode_num = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_map = (
            "NONE",
            "MOVE_LEFT",
            "MOVE_RIGHT",
        )
        self.n_actions = len(self.action_map)
        self.state_size = 205
        self.agent = DQN(self.state_size, self.n_actions, self.device)

        self.total_steps = 0
        self.epsilon = None

        self.prev_info = None

        self.state = None
        self.action = None

        self.summary_writer = SummaryWriter(
            log_dir="./ml/summary/" + time.strftime("%b%d_%Y_%H-%M-%S")
        )
        self.episode_loss = 0
        self.episode_reward = 0
        self.episode_len = 0

        self.reward_history = []
        self.best_mean = -100

    def reset(self):
        # Saving best model
        self.reward_history.append(self.episode_reward)
        mean_reward = np.mean(self.reward_history[-30:])
        if mean_reward > self.best_mean:
            self.agent.save_model("best.pt", self.total_steps)
            print(
                f"Mean reward updated {self.best_mean:.3f} -> {mean_reward:3f}, model saved.",
            )
            self.best_mean = mean_reward

        # Saving tensorboard summary
        self.summary_writer.add_scalar(
            "Loss", self.episode_loss / self.episode_len, self.episode_num
        )
        self.summary_writer.add_scalar("Reward", self.episode_reward, self.episode_num)
        self.summary_writer.add_scalar("Epsilon", self.epsilon, self.episode_num)
        self.summary_writer.flush()

        self.episode_num += 1

        self.prev_info = None

        self.state = None
        self.action = None

        self.episode_loss = 0
        self.episode_reward = 0
        self.episode_len = 0

    def get_state(self, scene_info):
        state = np.full(self.state_size, -1)

        # Platform's x coordinate
        state[0] = scene_info["platform"][0]

        # Ball's coordinate
        state[1:3] = scene_info["ball"]

        # Ball's movement
        if self.prev_info is not None:
            state[3:5] = (
                scene_info["ball"][0] - self.prev_info["ball"][0],
                scene_info["ball"][1] - self.prev_info["ball"][1],
            )
        else:
            state[3:5] = (0, 0)

        # Positions of all the remaining bricks
        bricks_pos = np.array(scene_info["bricks"]).flatten()
        state[5 : 5 + bricks_pos.size] = bricks_pos

        return state

    def get_reward(self, scene_info):
        reward = 0

        # Incentivize clearing bricks
        if self.prev_info is not None:
            reward += len(self.prev_info["bricks"]) - len(scene_info["bricks"])

        # Incentivize game completion
        if scene_info["status"] == "GAME_PASS":
            reward += 100

        elif scene_info["status"] == "GAME_OVER":
            # Penalize game fail
            reward += -100

        return reward

    def update(self, scene_info, *args, **kwargs):
        new_state = self.get_state(scene_info)

        # Serve the ball
        if not scene_info["ball_served"]:
            self.prev_info = scene_info
            return "SERVE_TO_LEFT"

        reset = "RESET" if scene_info["status"] in ("GAME_OVER", "GAME_PASS") else None
        if self.state is not None:
            self.reward = self.get_reward(scene_info)
            self.episode_reward += self.reward

            # Store transition into experience replay buffer
            self.agent.store_transition(
                self.state,
                self.action,
                self.reward,
                False if reset is None else True,
                new_state,
            )

            # Update model's parameter
            if len(self.agent.memory) > 4 * self.agent.batch_size:
                self.episode_loss += self.agent.learn()

        if reset is None:
            self.state = new_state
            self.epsilon = get_epsilon(self.total_steps)
            self.action = self.agent.choose_action(new_state, self.epsilon)

            self.prev_info = scene_info

        self.episode_len += 1
        self.total_steps += 1
        if self.total_steps % 10000 == 0:
            # Saving models every 10000 steps
            self.agent.save_model(
                f"checkpoint_{self.total_steps:06d}.pt", self.total_steps
            )
            print("Checkpoint model saved.")

        return reset or self.action_map[self.action]
