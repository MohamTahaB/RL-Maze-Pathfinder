import random

import numpy as np

from gym_examples.envs import GridWorldEnv


def main() :
    env = GridWorldEnv()
    state_size = env.size**2
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.7
    discount_rate = 0.7
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 1000
    max_steps = 120  # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        statei = state["agent"][0]*5 + state["agent"][1]
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()


            else:
                # exploit
                action = np.argmax(qtable[statei, :])

            # take action and observe reward
            new_state, reward, done, info = env.step(action)
            new_statei = new_state["agent"][0]*5 + new_state["agent"][1]

            # Q-learning algorithm
            qtable[statei, action] = qtable[statei, action] + learning_rate * (
                        reward + discount_rate * np.max(qtable[new_statei, :]) - qtable[statei, action])

            # Update to our new state
            state = new_state
            statei = state["agent"][0]*5 + state["agent"][1]

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)


    #print(qtable)

    print(qtable)
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    statei = state["agent"][0]*5 + state["agent"][1]
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[statei, :])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state
        statei = state["agent"][0] * 5 + state["agent"][1]

        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()