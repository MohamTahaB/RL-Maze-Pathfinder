import gym
from gym import spaces
import numpy as np
import random


class QLearningAgent:

    def __init__(self, env ,learning_rate =0.9, discount_rate=0.8 ,  epsilon=1 , decay_rate=0.005):
        # hyper parameters
        self.state = None
        self.learning_rate ,self.discount_rate , self.epsilon , self.decay_rate = learning_rate , discount_rate ,  epsilon , decay_rate

        self.action_space = env.action_space

        self.qtable = np.zeros((env.nb_states, env.nb_actions))

    def learn(self, num_episodes, max_steps, env, state):

        for episode in range(num_episodes):

            self.state = state

            for s in range(max_steps):

                action = self.choose_action()  # choose an action according to the explore/exploit policy

                observation, reward, done, info = env.step(action)
                new_state = env.state_table.index(list(observation['agent']))

                # Q-learning algorithm
                self.qtable[self.state, action] = self.qtable[self.state, action] + self.learning_rate * (
                        reward + self.discount_rate * np.max(self.qtable[new_state, :]) - self.qtable[self.state, action])

                #env.render()
                self.state = new_state

                # if done, finish episode
                if done :
                    # On reset l'environnement
                    observation = env.reset()
                    break

            # Decrease epsilon
            self.decrease_epsilon(episode)

    def decrease_epsilon(self, episode):
        self.epsilon = np.exp(-self.decay_rate * episode)

    def choose_action(self):

        # exploration-exploitation tradeoff
        if random.uniform(0, 1) < self.epsilon:
            # explore
            action = self.action_space.sample()
        else:
            # exploit
            action = np.argmax(self.qtable[self.state, :])

        return action
