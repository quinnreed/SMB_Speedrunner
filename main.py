import argparse
import numpy as np
import retro
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
import nn

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
# print([env.unwrapped.get_action_meaning(action) for action in range(env.action_space.n)])
# print(observations[8192:].sum())

INPUT_SHAPE = (2048, )
input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE

class SMBProcessor(Processor):
    def process_observation(self, observation):
        processed_observation = observation[:INPUT_SHAPE[0]]

        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

model = nn.create_model(input_shape, nb_actions)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = SMBProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
                            value_max=1., value_min=.1, 
                            # value_max=.1, value_min=.1,
                            value_test=.05, nb_steps=10000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
            memory=memory, processor=processor, nb_steps_warmup=50000, 
            gamma=.99, target_model_update=10000, train_interval=4, 
            delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    weights_filename = 'dqn_smb1.h5f'
    checkpoint_weights_filename = 'checkpoint_weights/dqn_smb1_{step}.h5f'
    log_filename = 'dqn_smb1_log.json'
    callbacks = [
        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000),
        FileLogger(log_filename, interval=100)
    ]

    check_weights_file = tf.train.latest_checkpoint('./checkpoint_weights')
    if check_weights_file:
        dqn.load_weights(check_weights_file)

    dqn.fit(env, callbacks=callbacks, nb_steps=3000000, log_interval=10000, visualize=False)

    dqn.save_weights(weights_filename, overwrite=True)

    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_smb1.h5f'

    dqn.load_weights(weights_filename)

    dqn.test(env, nb_episodes=10, visualize=True)