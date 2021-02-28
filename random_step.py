"""
Test with random steps. Used to just make sure the emulator is working, the rom is connected appropriately
"""
import retro
from time import sleep
from PIL import Image, ImageOps
import numpy as np

def main():
    env = retro.make(game='SuperMarioBros-Nes', use_restricted_actions=retro.Actions.DISCRETE, 
                    ) #  obs_type=retro.Observations.RAM)
    obs = env.reset()
    frame = 0
    while True:
        frame += 1
        action = 6
        # action = env.action_space.sample()
        # print(action)
        obs, rew, done, info = env.step(action)

        player_pos = env.get_ram()[941]

        max_x = env.get_screen().shape[1]
        min_x = 0

        left_x = player_pos - 32
        if left_x < 0:
            left_x = 0

        right_x = player_pos + 112
        if right_x > max_x:
            right_x = max_x

        viz_obs = env.get_screen()[:, left_x:right_x, :]

        viz_obs = ImageOps.grayscale(Image.fromarray(viz_obs))

        viz_obs = viz_obs.resize((12, 20))

        viz_obs.show()

        # print("frame:", frame, ":", rew, -1, rew - 1)

        # print(info)


        env.render()

        sleep(0.0166)

        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()