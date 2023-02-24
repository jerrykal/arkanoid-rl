import os
import sys

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(root_folder)

import pickle
import time

import numpy as np
import pygame
from mlgame.game.generic import quit_or_esc
from mlgame.view.view import PygameView

from src.game import Arkanoid


class MLPlay:
    def __init__(self, level, *args, **kwargs):
        self.level = level

        self.data = {"features": [], "targets": []}

        self.landing_x = None
        self.landing_frame = 0
        self.command_seq = ["SERVE_TO_RIGHT"]

        self.prev_ball_pos = None

    def reset(self):
        return

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
        if scene_info["status"] == "GAME_PASS":
            self.flush_to_file()
            return "QUIT"

        frame = scene_info["frame"]
        ball_pos = scene_info["ball"]

        if frame > len(self.command_seq) and ball_pos[1] >= 395:
            self.landing_x = 5 * (ball_pos[0] // 5)
            self.landing_frame = frame
            return "RESET"

        platform_center_x = scene_info["platform"][0] + 20

        if len(self.command_seq) <= frame <= self.landing_frame:
            state = self.get_state(scene_info)
            self.record(state, self.landing_x)

            if platform_center_x > self.landing_x:
                self.command_seq.append("MOVE_LEFT")
            elif platform_center_x < self.landing_x:
                self.command_seq.append("MOVE_RIGHT")
            else:
                self.command_seq.append("NONE")

        self.prev_ball_pos = scene_info["ball"]

        command = self.command_seq[frame] if frame < len(self.command_seq) else "NONE"
        return command

    def record(self, x, y):
        self.data["features"].append(x)
        self.data["targets"].append(y)

    def flush_to_file(self):
        filename = time.strftime(f"level_{self.level:02d}") + ".pickle"
        if not os.path.exists(os.path.dirname(__file__) + "/log"):
            os.makedirs(os.path.dirname(__file__) + "/log")
        filepath = os.path.join(os.path.dirname(__file__), "./log/" + filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.data, f)


def main():
    pygame.init()

    for level in range(24):
        game = Arkanoid(difficulty="EASY", level=level + 1)
        ai_name = game.ai_clients()[0]["name"]

        mlplay = MLPlay(level=level + 1)

        scene_init_info_dict = game.get_scene_init_data()
        game_view = PygameView(scene_init_info_dict)

        quit = False
        while not quit:
            scene_info = game.get_data_from_game_to_player()[ai_name]
            commands = {ai_name: mlplay.update(scene_info)}

            if commands[ai_name] == "QUIT":
                break

            if commands[ai_name] == "RESET":
                mlplay.reset()
                game.reset()
                game_view.reset()
                continue

            game.update(commands)

            game_progress_data = game.get_scene_progress_data()
            game_view.draw(game_progress_data)

            quit = quit_or_esc()

        if quit:
            break


if __name__ == "__main__":
    main()
