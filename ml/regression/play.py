import os
import pickle

import numpy as np


class MLPlay:
    def __init__(self, *args, **kwargs):
        self.ball_served = False
        self.prev_ball_pos = (0, 0)

        with open(
            os.path.join(os.path.dirname(__file__), "save", "model.pickle"), "rb"
        ) as f:
            self.model = pickle.load(f)

    def get_state(self, scene_info):
        state = np.full(204, -1)
        state[:2] = scene_info["ball"]
        state[2:4] = (
            scene_info["ball"][0] - self.prev_ball_pos[0],
            scene_info["ball"][1] - self.prev_ball_pos[1],
        )

        bricks_pos = np.concatenate(
            (
                np.array(scene_info["bricks"]).flatten(),
                np.array(scene_info["hard_bricks"]).flatten(),
            )
        )
        state[4 : 4 + bricks_pos.size] = bricks_pos

        return state

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] == "GAME_OVER" or scene_info["status"] == "GAME_PASS":
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_RIGHT"
        else:
            state = self.get_state(scene_info)
            state = state[np.newaxis, ...]

            predicted_landing = self.model.predict(state)

            platform_center_x = scene_info["platform"][0] + 20
            if platform_center_x < predicted_landing:
                command = "MOVE_RIGHT"
            elif platform_center_x > predicted_landing:
                command = "MOVE_LEFT"
            else:
                command = "NONE"

        self.prev_ball_pos = scene_info["ball"]
        return command

    def reset(self):
        self.ball_served = False
