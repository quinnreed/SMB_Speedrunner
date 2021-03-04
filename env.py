import numpy as np
from PIL import Image
import gym
from retro import RetroEnv

NES_BYTES = 2048
DEFAULT_LOC = 40
SHAPE = (2588,)

class MarioEnv(RetroEnv):
    def __init__(self, *args, **kwargs):
        super(MarioEnv, self).__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=SHAPE, dtype=np.uint8)

        self.frame_count = 0
        self.last_loc = DEFAULT_LOC

    def _update_obs(self):
        """
        Overwriting this function
        """
        self.ram = self.get_ram()
        self.img = self.get_screen()

        ram = self.ram[:NES_BYTES]

        player_pos = ram[941]

        max_x = self.img.shape[1]
        min_x = 0

        left_x = player_pos - 32
        if left_x < 0:
            left_x = 0

        right_x = player_pos + 112
        if right_x > max_x:
            right_x = max_x

        player_y = ram[206]

        top_y = player_y - 100
        if ram[181] == 0 or top_y < 0:
            top_y = 0

        bottom_y = player_y + 100
        if ram[181] > 1:
            bottom_y = self.img.shape[0]
            top_y = bottom_y - 100

        viz_obs = self.img[top_y:bottom_y, left_x:right_x, :]

        # print(viz_obs.shape, top_y, bottom_y, left_x, right_x)

        viz_obs = Image.fromarray(viz_obs)

        viz_obs = np.asarray(viz_obs.resize((12, 15))).flatten()

        return np.concatenate((ram, viz_obs))

    def compute_step(self):
        if self.players > 1:
            reward = [self.data.current_reward(p) for p in range(self.players)]
        else:
            reward = self.data.current_reward()
        done = self.data.is_done()
        info = self.data.lookup_all()

        reward = np.clip(reward, 0, 1)

        reward -= 1/60 # subtract the frame

        if reward <= 0 and self.ram[941] == self.last_loc:
            self.frame_count += 1
        else:
            self.last_loc = self.ram[941]
            self.frame_count = 0

        if (self.frame_count >= 600 or info['lives'] <= 1) and self.ram[1904] != 2:
            done = True
            self.frame_count = 0
            self.last_loc = DEFAULT_LOC
            reward = -1000
        elif self.ram[1904] == 2:
            done = True
            self.frame_count = 0
            self.last_loc = DEFAULT_LOC
            reward = 1000

        return reward, done, info