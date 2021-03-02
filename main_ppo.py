from time import sleep
import argparse
import numpy as np
import retro
from env import MarioEnv
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()

env = make_vec_env(MarioEnv, n_envs=1, env_kwargs={'game': 'SuperMarioBros-Nes', 
                'use_restricted_actions': retro.Actions.ALL, 'obs_type': retro.Observations.RAM, 'state': None})
nb_actions = env.action_space.n

policy = MlpLstmPolicy

model = PPO2(policy, env, verbose=1, nminibatches=1)

weights_filename = 'ppo2-firstpass'
if args.mode == 'train':
    callbacks = [
        CheckpointCallback(500000, save_path='./checkpoint_weights/ppo2-firstpass/', name_prefix='ppo2-firstpass'),
    ]

    model.load(weights_filename)
    model.learn(2000000, callback=callbacks, log_interval=1000, tb_log_name="ppo2-firstpass")    
    model.save('ppo2-firstpass')


elif args.mode == 'test':
    model.load(weights_filename)

    # # dqn.test(env, nb_episodes=10, visualize=True)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
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