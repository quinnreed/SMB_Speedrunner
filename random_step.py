"""
Test with random steps. Used to just make sure the emulator is working, the rom is connected appropriately
"""
import retro

def main():
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()