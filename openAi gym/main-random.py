import random

import gym
import numpy as np

import gym_examples
from gym_examples.envs import GridWorldEnv


def main() :
    env = GridWorldEnv()
    state = env.reset()

    num_steps = 99
    done = True
    while  done:
        for s in range(num_steps + 1):
            print(f"step: {s} out of {num_steps}")

            # sample a random action from the list of available actions
            action = env.action_space.sample()

            # perform this action on the environment
            observation, reward, done, info = env.step(action)

            # print the new state
            env.render()
            if done:
                break
        if done:
            state = env.reset()


    # end this instance of the taxi environment
    env.close()

if __name__ == "__main__":
    main()