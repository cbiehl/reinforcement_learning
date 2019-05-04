"""
    Natural Policy Gradient (TRPO)
"""

import os
import sys
import pickle
import gym
import quanser_robots
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal

#seed = 42
#log_interval = 1
#
#env = gym.make('THE_ENVIRONMENT')
#env.seed(seed)
#torch.manual_seed(seed)

def initialize_params(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.001)


def flatten_gradient(gradients):
    flat_gradients = []
    for g in gradients:
        flat_gradients.append(g.view(-1))

    return torch.cat(flat_gradients)


def flatten_params(model):
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1))

    return torch.cat(params)


def get_kl(new_actor, old_actor, states):
    mean, std = new_actor(torch.Tensor(states))
    logstd = torch.log(std)
    mean2, std2 = old_actor(torch.Tensor(states))
    mean2 = mean2.detach()
    logstd2 = torch.log(std2).detach()
    std2 = std2.detach()

    kl = logstd2 - logstd + (std2.pow(2) + (mean2 - mean).pow(2)) / (2 * std.pow(2)) - 0.5

    return kl.sum(1, keepdim=True)


class Policy(nn.Module):
    def __init__(self, n_state_dims, n_actions, initialize_params=None):
        super(Policy, self).__init__()
        self.n_actions = n_actions
        self.affine1 = nn.Linear(n_state_dims, 64)
        self.mean = nn.Linear(64, n_actions)
        self.sigma = nn.Linear(64, n_actions)

        self.rewards = []
        self.states = []
        self.state_values = []
        self.log_probs = []

        if initialize_params:
            self.apply(initialize_params)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        mean = self.mean(x)
        sigma =self.sigma(x)
        sigma = F.softplus(sigma)

        return mean, sigma

    def update(self, new_params):
        i = 0
        for params in self.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[i: i + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            i += params_length


class Critic(nn.Module):
    def __init__(self, n_state_dims, optimizer=None, loss_fn=None, initialize_params=None):
        super(Critic, self).__init__()

        self.affine1 = nn.Linear(n_state_dims, 128)
        self.head = nn.Linear(128, 1)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = F.smooth_l1_loss

        if initialize_params:
            self.apply(initialize_params)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        values = self.head(x)

        return values

    def update(self, true_returns, predicted_values):
        loss = self.loss_fn(predicted_values, true_returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class NaturalGradientPolicy:
    def __init__(self, env, gamma=0.98):
        self.env = env
        self.gamma = gamma
        self.actor = Policy(n_state_dims=env.observation_space.low.shape[0],
                            n_actions=env.action_space.low.shape[0],
                            initialize_params=initialize_params)

        self.critic = Critic(n_state_dims=env.observation_space.low.shape[0],
                             initialize_params=initialize_params)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-2)

    def get_action(self, state, stochastic=True, store=True):
        if type(state) != torch.Tensor:
            state = torch.from_numpy(state).float()

        mean, sigma = self.actor(state)
        state_value = self.critic(state)
        lp = torch.Tensor([1.0])

        # using a multivariate gaussian did not improve the result here
#        cov = torch.zeros(mean.size()[0], mean.size()[0])
#        for i in range(mean.size()[0]):
#            cov[i,i] = 1.0 #sigma[i] + 0.001
#
#        d = MultivariateNormal(mean, cov)

        if stochastic:
            d = Normal(mean, sigma)
            action = d.sample()
            lp = d.log_prob(action).sum()

            if store:
                self.actor.log_probs.append(lp)
                self.actor.state_values.append(state_value)
                self.actor.states.append(state)

        else:
            action = mean.detach()

        return action.numpy(), lp

    def get_fisher_vector(self, loss_grad, states, rate=1e-2):
        kl = get_kl(self.actor, self.actor, states).mean()

        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        grads = torch.cat([grad.view(-1) for grad in grads])

        kl_grads = (grads * loss_grad).sum()
        grads2 = torch.autograd.grad(kl_grads, self.actor.parameters())
        flat_grad2_kl = torch.cat([grad.contiguous().view(-1) for grad in grads2]).detach()

        return flat_grad2_kl + loss_grad * rate

    def get_conjugate_gradient(self, g, states, n_steps, eps=1e-10):
        cg = torch.zeros(g.size())
        ng = g.clone()
        p = g.clone()
        gsquare = torch.dot(ng, ng)

        for i in range(n_steps):
            Fisherv = self.get_fisher_vector(p, states)

            alpha = gsquare / torch.dot(p, Fisherv)
            cg = cg + alpha * p

            ng = ng - (alpha * Fisherv)
            gsquare_p = torch.dot(ng, ng)
            beta = gsquare_p / gsquare
            p = ng + beta * p

            gsquare = gsquare_p

            if gsquare < eps:
                break

        return cg

    def linesearch(self, input_params, ng_step, expected_improvement, n_step=10, eps=1e-2):
        actor_loss = self.get_actor_loss()

        for i in range(n_step):
            step = i / 2
            params = input_params + step * ng_step
            self.actor.update(params)

            step_actor_loss = self.get_actor_loss()
            improvement = actor_loss - step_actor_loss
            ratio = improvement / (expected_improvement * step)

            if ratio > eps:
                return params

        return input_params

    def get_actor_loss(self):
        log_probs = []
        for state in self.actor.states:
            action, log_prob = self.get_action(state, store=False)
            log_probs.append(log_prob)


        actor_loss = (-self.actor.advantages * torch.stack(log_probs)).mean()

        return actor_loss.item()

    def optimize(self, reset_history=True, trust_region_bound=8e-3):
        states = torch.stack(self.actor.states)
        log_probs = torch.stack(self.actor.log_probs)
        returns = []
        R = 0
        for r in self.actor.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        self.actor.returns = torch.tensor(returns)

        v = self.critic(states)
        advantages = self.actor.returns - v.view(-1)
        self.actor.advantages = advantages

        actor_loss = (-advantages * log_probs).mean()
        loss_grad = torch.autograd.grad(actor_loss, self.actor.parameters(), create_graph=True)
        loss_grad = flatten_gradient(loss_grad).detach()

#        H = get_hessian(loss_grad, self.actor)
#        step = np.linalg.pinv(H) @ loss_grad.detach().numpy()

        step_dir = self.get_conjugate_gradient(-loss_grad.detach(),
                                               states,
                                               n_steps=10)

        nlr = torch.sqrt(trust_region_bound / \
                         0.5 * (step_dir.dot(self.get_fisher_vector(step_dir, states))))
        ng_step = nlr * step_dir

        expected_improvement = -loss_grad.dot(ng_step)

        last_params = flatten_params(self.actor)
        params = self.linesearch(last_params, ng_step, expected_improvement)
        self.actor.update(params)

        self.critic.update(self.actor.returns, v.view(-1))

        if reset_history:
            del self.actor.rewards[:]
            del self.actor.states[:]
            del self.actor.state_values[:]
            del self.actor.log_probs[:]
            del self.actor.returns
            del self.actor.advantages

    def train(self, n_episodes, output_file=None, render=False, logging=False, log_interval=1):
        for i_episode in count(1):
            if i_episode > n_episodes:
                if output_file != None:
                    with open(output_file, 'wb') as f:
                        pickle.dump(self, f)

                return

            state = self.env.reset()
            cumulative_reward = 0.0

            for t in range(1, 100000):
                action, _ = self.get_action(state, stochastic=True)

                # disable nasty env logging (otherwise - get ready for spam!)
                sys.stdout = open(os.devnull, "w")

                state, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()

                # restore console stdout
                sys.stdout = sys.__stdout__

                self.actor.rewards.append(reward)

                cumulative_reward += reward
                if done:
                    break

                self.optimize()

            if logging and i_episode % log_interval == 0:
                print()
                print('Episode {}\tCumulative reward: {:.2f}'.format(i_episode, cumulative_reward))
                print()
