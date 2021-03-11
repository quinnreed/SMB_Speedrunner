from time import sleep
import argparse
import numpy as np
import retro
from env import MarioEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()

env = make_vec_env(MarioEnv, n_envs=1, env_kwargs={'game': 'SuperMarioBros-Nes', 
                'use_restricted_actions': retro.Actions.FILTERED, 'obs_type': retro.Observations.RAM})
nb_actions = env.action_space.n

policy = MlpPolicy

model = PPO(policy, env, 
            learning_rate=2.5e-4, n_steps=128, batch_size=32, n_epochs=3, clip_range=0.1, ent_coef=.01, vf_coef=1, 
            verbose=1)

old_weights_filename = 'ppo-torch-mbool-xstep1-deathon'
new_weights_filename = 'ppo-torch-mbool-xstep1-deathon'
if args.mode == 'train':
    callbacks = [
        CheckpointCallback(500000, save_path=f'./checkpoint_weights/{new_weights_filename}/', name_prefix=new_weights_filename),
    ]

    model = PPO.load(old_weights_filename, env=env)
    model.learn(1000000, callback=callbacks, log_interval=5, tb_log_name=new_weights_filename)    
    model.save(new_weights_filename)


elif args.mode == 'test':
    model = PPO.load(old_weights_filename, env=env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        # print(action)
        obs, rewards, dones, info = env.step(action)

        print(rewards)
        env.render()

    # movie = retro.Movie('SuperMarioBros-Nes-Level1-1-000000.bk2')
    # movie.step()

    # env = retro.make(
    #     game=movie.get_game(),
    #     state=None,
    #     # bk2s can contain any button presses, so allow everything
    #     use_restricted_actions=retro.Actions.ALL,
    #     players=movie.players,
    # )
    # env.initial_state = movie.get_state()
    # env.reset()

    # while movie.step():
    #     keys = []
    #     for p in range(movie.players):
    #         for i in range(env.num_buttons):
    #             keys.append(movie.get_key(i, p))

    #     print(env.unwrapped.get_action_meaning(keys))

    #     env.step(keys)
    #     env.render()