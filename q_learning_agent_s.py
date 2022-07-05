import numpy as np
import random


class QLearningAgentS:

    def __init__(self, env, learning_rate=0.6, discount_rate=0.5, epsilon=0.8, decay_rate=0.005):
        # hyper parameters
        self.learning_rate = learning_rate  # alpha
        self.discount_rate = discount_rate  # gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.state = None

        self.action_space = env.action_space

        self.qtable = np.zeros((env.nb_states, env.nb_actions))

    def learn(self, num_episodes, max_steps, env, state):
        #Initializing the reward
        reward=0

        # Starting the SARSA learningaction1 = self.choose_action(self.state)
        for episode in range(num_episodes):
            t = 0
            self.state = env.state_table.index(list(env.reset()[0]['agent']))
            action1 = self.choose_action(self.state)

            while t < max_steps:



                #Visualizing the training
                #env.render()

                #Getting the next state
                observation, reward, done, info = env.step(action1)
                new_state = env.state_table.index(list(observation['agent']))

                #Choosing the next action
                action2 = self.choose_action(new_state)


                #Learning the Q-value

                self.update_qtable(self.state, new_state, reward, action1, action2)

                self.state = new_state
                action1 = action2

                #Updating the respective vaLues
                t += 1
                reward += 1

                #If at the end of learning process
                if done:
                    break


    def decrease_epsilon(self, episode):
        self.epsilon = np.exp(-self.decay_rate * episode)

    def choose_action(self, state):
        # exploration-exploitation tradeoff
        if random.uniform(0, 1) < self.epsilon:
            # explore
            action = self.action_space.sample()
        else:
            # exploit
            action = np.argmax(self.qtable[state, :])

        return action

    def update_qtable(self, state, state2, reward, action, action2):
        predict = self.qtable[state, action]
        target = reward + self.learning_rate * self.qtable[state2, action2]
        self.qtable[state, action] = self.qtable[state, action] + self.learning_rate* (target - predict)
