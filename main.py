from time import sleep
import argparse
import numpy as np
import retro
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, Callback
import nn
from PIL import ImageOps, Image

WINDOW_LENGTH = 4

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
# parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

env = retro.make(game='SuperMarioBros-Nes', use_restricted_actions=retro.Actions.DISCRETE, 
                 obs_type=retro.Observations.RAM)
observations = env.reset()
nb_actions = env.action_space.n

# print(env.action_space)
# print(np.unique([env.unwrapped.get_action_meaning(env.action_space.sample()) for _ in range(100)]))
# print([(env.unwrapped.get_action_meaning(action), action) for action in range(env.action_space.n)])
# print(observations[8192:].sum())


BYTES = 2048
INPUT_SHAPE = (BYTES + 540, )
input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE

class SMBProcessor(Processor):
    frame = 0
    alive = True
    last_location = 40

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)

        # sleep(0.0166)
        
        if reward <= 0 and env.get_ram()[941] == self.last_location :
            self.frame += 1
        else:
            self.last_location = env.get_ram()[941]
            self.frame = 0

        if (self.frame >= 300 or info['lives'] <= 1) and env.get_ram()[1904] != 2:
            # print(self.frame)
            reward = -5
            self.frame = 0
            self.last_location = 40
            done = True
        elif env.get_ram()[1904] == 2:
            done = True
            self.frame = 0
            self.last_location = 40
            reward = 1000

        return observation, reward, done, info

    def process_observation(self, observation):
        processed_observation = observation[:BYTES]

        player_pos = env.get_ram()[941]

        max_x = env.get_screen().shape[1]
        min_x = 0

        left_x = player_pos - 32
        if left_x < 0:
            left_x = 0

        right_x = player_pos + 112
        if right_x > max_x:
            right_x = max_x

        player_y = env.get_ram()[206]

        top_y = player_y - 100
        if env.get_ram()[181] == 0:
            top_y = 0

        bottom_y = player_y + 100
        if env.get_ram()[181] > 1:
            bottom_y = -1

        viz_obs = env.get_screen()[top_y:bottom_y, left_x:right_x, :]

        viz_obs = Image.fromarray(viz_obs)

        viz_obs = np.asarray(viz_obs.resize((12, 15))).flatten()

        new_obs = np.concatenate((processed_observation, viz_obs))

        assert new_obs.shape == INPUT_SHAPE
        return new_obs.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, 0, 1.) - 1/60 # for the frame

processor = SMBProcessor()

class SMBCallback(Callback):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def on_episode_begin(self, episode, logs):
        # print(self.processor.frame)
        self.processor.frame = 0
        # print(self.processor.frame)


model = nn.create_model(input_shape, nb_actions)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
                            value_max=.5, value_min=.3, 
                            # value_max=.5, value_min=.1,
                            value_test=.05, nb_steps=2000000)
                            # value_test=.05, nb_steps=5000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
            memory=memory, processor=processor, nb_steps_warmup=50000, 
            gamma=.99, target_model_update=10000, train_interval=4, 
            delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

# weights_filename = 'dqn_smb1_ram+viz1.h5f'
if args.mode == 'train':
    checkpoint_weights_filename = 'checkpoint_weights/ram+color_viz/dqn_smb1_ram+color_viz_{step}.h5f'
    log_filename = 'dqn_smb1_log.json'
    callbacks = [
        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000),
        FileLogger(log_filename, interval=100),
        SMBCallback(processor)
    ]

    # check_weights_file = tf.train.latest_checkpoint('checkpoint_weights')
    # check_weights_file = weights_filename
    # if check_weights_file:
    # dqn.load_weights(check_weights_file)

    # for _ in range(3):
    dqn.fit(env, callbacks=callbacks, nb_steps=2000000, log_interval=10000, visualize=False)
    dqn.save_weights('dqn_smb1_ram+color_viz.h5f', overwrite=True)
    dqn.test(env, nb_episodes=10, visualize=True)

elif args.mode == 'test':
    dqn.load_weights(weights_filename)

    dqn.test(env, nb_episodes=10, visualize=True)