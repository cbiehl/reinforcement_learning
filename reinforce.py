"""
Policy Gradient Algorithm: REINFORCE
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import gym
import pickle
import quanser_robots

import torch
from torch import nn
from torch import optim

import numpy as np


# -----------------------------------------------------------------------------
# Discretization
# -----------------------------------------------------------------------------

def discretize_action_space(env, n_actions=10, endpoint=True):
    """
    Discretizes the action space.

    :param env:                 environment
    :param n_actions:           number of actions
    """
    return np.linspace( \
       env.action_space.low[0], \
       env.action_space.high[0], \
       n_actions, endpoint=endpoint \
    )


# -----------------------------------------------------------------------------
# Neural network representing the policy
# -----------------------------------------------------------------------------

class Neural_Net():

    def __init__(self, n_inputs, n_outputs, n_hidden_neurons=16):
        """
        Constructor.

        :param env:                 environment
        :param n_inputs:            number of input neurons
        :paran n_outputs:           number of output neurons
        :param n_hidden_neurons:    number of hidden neurons
        """
        self.n_inputs = n_inputs
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = n_outputs

        # network definition
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden_neurons),
            nn.ReLU(),
            nn.Linear(self.n_hidden_neurons, self.n_outputs),
            nn.Softmax(dim=-1)
        )

        self.network.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


# -----------------------------------------------------------------------------
# Reinforce algorithm
# -----------------------------------------------------------------------------

class REINFORCE():
    """
    Reinforce policy gradient algorithm.
    """

    def __init__(self, env, n_actions=2, n_hidden_neurons=16):
        """
        Constructor.

        :param env:                 environment
        :param n_actions:           number of actions (for discretization)
        :param n_hidden_neurons:    number of hidden neurons (for neural net policy)
        """
        self.env = env
        self.n_actions = n_actions
        self.n_hidden_neurons = n_hidden_neurons
        self.action_space = discretize_action_space(self.env, self.n_actions, True) / 2

        self.policy = Neural_Net( \
            self.env.observation_space.shape[0], \
            self.n_actions, \
            self.n_hidden_neurons \
        )

        # define optimizer
        self.optimizer = optim.Adam(
            self.policy.network.parameters(),
            lr=5.0 # learning rate
        )


    def __train_network(self, batch):
        """
        Trains the network as soon as the batch is full.

        :param batch:               dictionary contraining states, actions, rewards
                                    used for training
        """
        print("Training network...")
        self.optimizer.zero_grad() # reset gradient to zero
        state_tensor = torch.FloatTensor(batch["states"])
        reward_tensor = torch.FloatTensor(batch["rewards"])
        # actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch["actions"])
        action_indices = torch.LongTensor(batch["actions"])

        # set action indices
        for i in range(self.n_actions):
            action_indices[np.where(action_tensor == self.action_space[i])] = i

        # calculate loss
        logprob = torch.log(
            self.policy.network(torch.FloatTensor(state_tensor))
        )

        selected_logprobs = -reward_tensor * \
            logprob[np.arange(len(action_indices)), action_indices]
        loss = selected_logprobs.sum()

        # calculate gradients
        loss.backward()
        # apply gradients and update parameters
        self.optimizer.step()


    def __get_prob_dist(self, s):
        """
        Gets the probability distribution over actions for state.

        :param s:                   current state
        :return:                    probability distribution over actions
        """
        return self.policy.network(torch.FloatTensor(s)).detach().numpy()


    def train(
        self,
        n_episodes=2000,
        n_batch=10,
        gamma=0.99):
        """
        Reinforce implementation (policy gradients).

        :param n_episodes:          number of episodes
        :param n_batch:             batch size
        :param gamma:               discount factor
        :return:                    total rewards
        """

        # set up lists to hold results
        total_rewards = []; b_rewards = []
        b_actions = []; b_states = []
        b_counter = 0

        # go over epsisodes
        for e in range(n_episodes):
            s = self.env.reset()
            states = []; rewards = []; actions = []
            done = False

            # while the episodes has not finished
            while not done:
                # select an action according to the probabilities
                # does not always choose the best action (exploration)
                a = np.random.choice( \
                    self.action_space, \
                    p=self.__get_prob_dist(s) \
                )
                # execute the action
                ns, r, done, _ = self.env.step(np.asarray([a]))

                states.append(s)
                rewards.append(r)
                actions.append(a)
                s = ns

                # we have reached a terminal state / end of trajectory
                if done:
                    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
                    # r - r.mean (baseline, reduces variance)
                    b_rewards.extend(r[::-1].cumsum()[::-1] - r.mean())
                    b_states.extend(states)
                    b_actions.extend(actions)
                    b_counter += 1
                    total_rewards.append(sum(rewards))

                    # update network as soon as we have
                    # collected enough data (batch is full)
                    if b_counter == n_batch:
                        self.__train_network({ \
                            "states": b_states, \
                            "rewards": b_rewards, \
                            "actions": b_actions \
                        })

                        # reset batch
                        b_states = []; b_rewards = []; b_actions = []
                        b_counter = 0

                    # print running average
                    print("\rEp: {} Average of last 10: {:.2f}".format( \
                        e + 1, np.mean(total_rewards[-10:])) \
                    )

        return total_rewards


    def predict(self, state):
        """
        Predicts an action for the given state.

        :param state:               state for which action should be predicted
        :return:                    action
        """
        action_probs = self.__get_prob_dist(state)
        # select the most probable action
        return np.argmax(action_probs)


    @staticmethod
    def load(path):
        """
        Loads a torch model.

        :param path:        path to pickled object
        :return:            loaded model
        """
        return pickle.load(open(path, "rb"))


    def save(self, path):
        """
        Saves dqn model.

        :param path:        path of saved object
        """
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()


## create environment
#env = gym.make("THE_ENVIRONMENT")
#s = env.reset()
#
#gamma = 0.99
#n_actions = 8
#n_episodes = 500
#
#n_batch = 1
#n_hidden_neurons = 8
#
## TRAINING
## -----------------------------------------------------------------------------
## apply reinforce algorithm
#RF = REINFORCE(env=env, n_actions=n_actions, n_hidden_neurons=n_hidden_neurons)
#RF.train(n_episodes=n_episodes, n_batch=n_batch, gamma=gamma)
#RF.save("./reinforce_model.pt")
#
## Render sample episode
## -----------------------------------------------------------------------------
#done = False
#sum_reward = 0.00
#s = env.reset()
#
#while not done:
#    a = RF.predict(s)
#    # execute the action
#    s, r, done, _ = env.step(np.asarray([a]))
#    sum_reward += r
#
#print("\nAccumulated reward: ", sum_reward)
