"""
    Deep Q-Network
"""

import torch
import pickle
import numpy as np
#import random

#from collections import deque

class DQN:
    """
    This class implements a deep q-network.
    """

    def __init__(
        self,
        state_size,
        actions,
        hidden_dim,
        hidden_activation=torch.nn.ReLU,
        learning_rate=0.001
    ):
        """
        Constructor.

        :param state_size:          number of states
        :param actions:             possible actions
        :param hidden_dim:          array of hidden dimensions
        :param hidden_activation:   activation functions
        :param learning rate:       learning rate alpha
        """
        self.state_size = state_size
        self.actions = actions
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate

        # discount rate
        self.gamma = 0.999
        # exploration rate
        self.epsilon = 0.90
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # number of iterations after which target network is replaced
        self.target_copy_interval = 250
        self.target_copy_cnt = 0

        # priority sampling params
        self.alpha=1.00
        self.beta = 0.40
        self.beta_max = 1.00
        self.beta_inc = 1.000005

        # size of replay memory
        self.memory = NaivePrioritizedBuffer(capacity=100000, prob_alpha=self.alpha) #deque(maxlen=10e6)

        # create the model specified
        self._build_model()


    def _build_model(self):
        """
        Neural network for q-learning.
        """
        # input layer
        layers = [torch.nn.Linear(self.state_size, self.hidden_dim[0]), self.hidden_activation()]
        # hidden layers
        for d in range(len(self.hidden_dim) - 1):
            layers.append(torch.nn.Linear(self.hidden_dim[d], self.hidden_dim[d+1]))
            layers.append(self.hidden_activation())
        # output layer
        layers.append(torch.nn.Linear(self.hidden_dim[-1], self.actions.shape[0]))

        # create model
        self.model = torch.nn.Sequential(*layers)
        # create target model
        self.targetmodel = torch.nn.Sequential(*layers)

        # initialize optimizer for backpropagation
        self.optimizer = torch.optim.Adam( \
            self.model.parameters(), \
            lr=self.learning_rate \
#            momentum=0.95 \
        )

        # initialize network weights
        def init_weights(l):
            if type(l) == torch.nn.Linear:
                torch.nn.init.xavier_uniform(l.weight)
                l.bias.data.fill_(0.01)

        self.model.apply(init_weights)
        self.targetmodel.load_state_dict(self.model.state_dict())


    def remember(self, state, action, reward, next_state, done):
        """
        Adds transition to the replay memory.

        :param state:       current state
        :param action:      action taken
        :param reward:      reward received
        :param next_state:  new state
        :param done:        flag indicating the end of the episode
        """
#        self.memory.append((state, action, reward, next_state, done))
        self.memory.push(state, action, reward, next_state, done)


    def get_action(self, state):
        """
        Determines the next action based on the current state.

        :param state:       current state
        :return:            best action of random action with probability epsilon
        """
        if np.random.rand() <= self.epsilon:
            # return random action
            return np.random.choice(self.actions)

        # else: use neural network to determine action
        q_values = self.model(torch.FloatTensor(state)).detach().numpy()
        action = self.actions[np.argmax(q_values)]

        return action


    def max_q_value(self, state):
        """
        Get the best q value over all actions for the current state.

        :param state:       current state
        :return:            best q value
        """
        q_values = self.targetmodel(torch.FloatTensor(state)).detach().numpy()

        return np.max(q_values)


    def replay(self, batch_size):
        """
        (Re-)train the neural network with new data.

        :param batch_size:  size of the batch used for training
        :return:            current loss
        """

# code for usage without priority sampling
# -----------------------------------------------------------------------------
#        # sample random minibatch from memory
#        minibatch = random.sample(self.memory, batch_size)
#        states = np.zeros((len(minibatch), self.state_size))
#        targets = np.zeros((len(minibatch), 1))
#
#        # train dqn agent with minibatch
#        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
#            self.target_copy_cnt += 1
#
#            if self.target_copy_cnt > self.target_copy_interval:
#                # update target network
#                self.targetmodel.load_state_dict(self.model.state_dict())
#                self.target_copy_cnt = 0
#
#            states[i, :] = state
#
#            if done:
#                # if episode ends: take immediate reward
#                # since there is no next state
#                targets[i, :] = reward
#            else:
#                # predict future discounted reward
#                targets[i, :] = reward + self.gamma * self.max_q_value(next_state)
#
#        # train the neural net with the state and target
#        X = torch.FloatTensor(states)
#        y = torch.FloatTensor(targets)
#
#        # forward pass
#        y_pred = torch.max(self.model(X), dim=1)[0]
#
#        loss = torch.nn.functional.smooth_l1_loss(y_pred, y.view(-1))
#
#        self.optimizer.zero_grad()
#        # backward pass / compute gradients
#        loss.backward()
#
#        self.memory.update_priorities(indices, prios.data.cpu().numpy())
#
#        # clip error
#        for param in self.model.parameters():
#            param.grad.data.clamp_(-1, 1)
#
#        # update parameters
#        self.optimizer.step()
#
#        # decay epsilon
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
#
#        return loss
# -----------------------------------------------------------------------------
# end of code for usage without priority sampling

        # update target network after number of iterations specified
        self.target_copy_cnt += 1
        if self.target_copy_cnt > self.target_copy_interval:
            self.targetmodel.load_state_dict(self.model.state_dict())
            self.target_copy_cnt = 0

        state, action, reward, next_state, done, indices, weights = self.memory.sample(batch_size, self.beta)
        targets = reward + self.gamma * self.max_q_value(next_state) * (1 - np.asarray(done))

        X = torch.FloatTensor(state)
        y = torch.FloatTensor(targets)

        # forward pass
        y_pred = torch.max(self.model(X), dim=1)[0]

        # calculate loss
        loss = (y.view(-1) - y_pred).pow(2) * torch.tensor(weights)
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        # backward pass / compute gradients
        loss.backward()

        self.memory.update_priorities(indices, prios.data.cpu().numpy())

        # clip error
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        # update parameters
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print("Epsilon:", self.epsilon)

        # increment beta
        if self.beta < self.beta_max:
            self.beta *= self.beta_inc
        print("Beta:", self.beta)

        return loss


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


# source: https://arxiv.org/pdf/1511.05952.pdf
# source: https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
class NaivePrioritizedBuffer(object):
    """
    This class implements a prioritized buffer which uses
    weights for sampling from the replay memory.
    """

    def __init__(self, capacity, prob_alpha=0.8):
        """
        Constructor.

        :param capacity:    size of the replay memory
        :param prob_alpha:  trades off uniform sampling and prioritized sampling
                                alpha = 0: uniform sampling
                                alpha = 1: prioritized sampling
        """
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)


    def push(self, state, action, reward, next_state, done):
        """
        Adds experience to the replay memory buffer.

        :param state:       current state
        :param action:      action taken
        :param reward:      reward received
        :param next_state:  next state
        :param done:        flag indicating the end of the episode
        """
        state = np.asarray(state)
        next_state = np.asarray(next_state)

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            # if there is still space: add element to buffer
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # replace element in buffer
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        # new examples should have maximal priority in order to guarantee
        # that each example is seen at least once
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity


    def sample(self, batch_size, beta=0.4):
        """
        Samples elements from the replay memory.

        :param batch_size:  size of the batch to sample
        :param beta:        bias annealing
        :return:            batch for training
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights


    def update_priorities(self, batch_indices, batch_priorities):
        """
        Updates the priorities.

        :param batch_indices:
        :param batch_priorities:
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


    def __len__(self):
        """
        Calculates the size of the buffer.

        :return:        size of buffer
        """
        return len(self.buffer)
