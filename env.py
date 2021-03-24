import numpy as np
from PIL import Image
import gym
from retro import RetroEnv
import logging
import datetime as dt
import csv

envlog = logging.getLogger('env')
envlog.setLevel(logging.DEBUG)
fh = logging.FileHandler('./env.log')
envlog.addHandler(fh) 
ch = logging.StreamHandler()
envlog.addHandler(ch)

NES_BYTES = 2048
DEFAULT_LOC = 40
SHAPE = (2588,)

class MarioEnv(RetroEnv):
    def __init__(self, *args, **kwargs):
        super(MarioEnv, self).__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=SHAPE, dtype=np.uint8)

        self.frame_count = 0
        # self.last_loc = {'0xA690': DEFAULT_LOC}
        self.last_loc = {}
        self.world_frames = {}
        self.current_level = 0 # 0 = '1'
        self.current_world = 0 # 0 = '1'
        self.coords = []
        self.stasis_loc = 0
        self.stasis_pix = []

    def prep_episode(self):
        self.frame_count = 0
        self.last_loc = {}
        self.world_frames = {}
        self.current_level = 0 # 0 = '1'
        self.current_world = 0 # 0 = '1'
        self.coords = []
        self.stasis_loc = 0
        self.stasis_pix = []

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

        level_addr = hex(np.array(self.ram[231:233]).view('<u2')[0])

        # print(level_addr)

        reward = new_loc = self.ram[109] * 256 + self.ram[134] ## Ram x6D * 256 + x86. This is x position in level including the screen

        # if level_addr in self.last_loc:
        #     reward -= self.last_loc[level_addr]
        # else:
        #     reward = 0

        # self.stasis_pix += [new_loc]

        pix_diff = 0
        if level_addr in self.last_loc:
            pix_diff = new_loc - self.last_loc[level_addr] # - self.stasis_loc # self.last_loc[level_addr]

        if level_addr in self.world_frames:
            # print(reward, self.world_frames[level_addr])
            
            reward /= self.world_frames[level_addr]            

            # reward += new_loc / self.world_frames[level_addr]
        else:
            reward = 0
            self.world_frames[level_addr] = 0

        if self.ram[14] != 0x00:
            self.last_loc[level_addr] = new_loc
        else:
            reward = 0

        for lev in self.world_frames.keys():
            self.world_frames[lev] += 1

        self.coords += [new_loc, self.ram[0xCE]]

        reward -= (1/60) # * self.frame_count # subtract the frame
        # reward -= 1

        # if np.std(self.stasis_pix) < 1 and self.ram[14] == 0x08: #and self.ram[941] == self.last_loc[level_addr]:
        if pix_diff <= 0 and self.ram[14] == 0x08: #and self.ram[941] == self.last_loc[level_addr]:
        # if reward <= 0 and self.ram[14] == 0x08: #and self.ram[941] == self.last_loc[level_addr]:
            self.frame_count += 1
        else:
            # self.last_loc = self.ram[941]
            self.frame_count = 0
            self.stasis_loc = new_loc
            self.stasis_pix = [new_loc]

        if (self.frame_count >= 600 or info['lives'] <= 1) and self.ram[1904] != 2:
            done = True
            self.prep_episode()
            reward += -100
        elif info['levelLo'] != self.current_level or info['levelHi'] != self.current_world:
            envlog.info(f"IT WON {self.current_world}-{self.current_level}, {dt.datetime.now()}")
            self.current_level = info['levelLo']
            self.current_world = info['levelHi']
            done = True
            self.prep_episode() # this will be removed later TODO
            reward += 10000

        if done:
            with open('locs.csv', 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.coords)
            self.coords = []

        return reward, done, info