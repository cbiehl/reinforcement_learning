import math
import pickle
import numpy as np
from collections import deque
from scipy.sparse.linalg import lsqr
from scipy.linalg import lstsq
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from featurizer import RBFFeatureGenerator, ApproximateRBFGenerator, FourierFeatureGenerator


class LSPI:
    def __init__(self, env, actions, action_bins, n_basisfunctions=10, sigma=1.0, damp=0.0):
        self.env = env
        self.actions = actions
        self.action_bins = action_bins
        self.action_dim = env.action_space.low.shape[0]
        self.state_dim = env.observation_space.low.shape[0]

        self.basis_functions = ApproximateRBFGenerator(action_bins, n_basisfunctions, normalize=True)
#        self.basis_functions = RBFFeatureGenerator(action_bins, n_basisfunctions, normalize=False)
#        self.basis_functions = FourierFeatureGenerator(actions, n_basisfunctions, nu='auto')

        self.nbf = n_basisfunctions
        self.damp = damp
        self.w = np.random.uniform(-1.0, 1.0, size=(self.nbf * self.actions.shape[0]))


    def get_action(self, state):
        # calculate q values for all actions and take argmax
        best_action = None
        best_q = -np.inf

        for i in range(self.actions.shape[0]):
            phi = self.basis_functions.transform(state, self.actions[i]).reshape(-1)
            q = self.w.T.dot(phi)
            if q > best_q:
                best_action = self.actions[i]
                best_q = q

        return np.asarray([best_action])


    def sample_from_policy(self, n_samples=1000, state=None, e=0.1):
        samples = []

        for i in range(n_samples):
            action = None
            if np.random.rand() <= e:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(state)

            next_state, reward, done, info = self.env.step(action)
            samples.append(np.concatenate((state, action, next_state, [reward])))

            if done:
                state = self.env.reset()
            else:
                state = next_state

        return np.asarray(samples)


    def lstdq(self, samples, gamma):
        A = np.identity(self.nbf * self.actions.shape[0]) * 0.0001
        b = np.zeros(self.nbf * self.actions.shape[0])

        for sample in samples:
            # samples: s, a, s', r
            state = sample[0:self.state_dim]
            action = sample[self.state_dim]
            next_state = sample[self.state_dim + self.action_dim:self.state_dim * 2 + self.action_dim]
            reward = sample[-1]

            phi = self.basis_functions.transform(state, action).reshape(-1)
            best_action = self.get_action(next_state)
            phi_prime = self.basis_functions.transform(next_state, best_action).reshape(-1)

            A = A + np.outer(phi, (phi - gamma * phi_prime))
            b = b + phi * reward

        self.last_w = np.copy(self.w)

        if np.linalg.matrix_rank(A) == A.shape[0]:
            self.w = np.linalg.solve(A, b)
        else:
            self.w = lstsq(A, b)[0]

#       self.w = lsqr(A, b, damp=self.damp)[0]


    def fit(self, samples, max_iter=100, epsilon=1e-4, gamma=0.99):
        self.basis_functions.fit(samples[:, 0:self.state_dim])
        last_threshold = 1e10
        threshold = 1e10
        not_changed = 0
        i = 0

        while i < max_iter and epsilon < threshold and not_changed < 5:
            self.lstdq(samples, gamma)

            threshold = np.linalg.norm(self.w - self.last_w)
            print("Norm ||w - w'||:", threshold)
            print("Weights:", self.w)
            print()

            if np.allclose(threshold, last_threshold, atol=1e-5) or threshold > last_threshold:
                not_changed += 1
            else:
                not_changed = 0

            last_threshold = threshold
            i = i + 1

        print("Done with LSPI iteration, final weights:")
        print(self.w)
        print()


    def fit_online(self, samples_, max_iter=100, epsilon=1e-4, gamma=0.99):
        self.basis_functions.fit(samples_[:, 0:self.state_dim])
        samples = deque(maxlen=20000)
        for sample in samples_:
            samples.append(sample)

        e = 1.0
        e_min = 0.01
        e_decay = 0.995
        update_rate = 200
        threshold = np.inf
        i = 0
        j = 0
        s = self.env.reset()

        while i < max_iter and epsilon < threshold:
            A = np.identity(self.nbf * self.actions.shape[0]) * 0.0001
            b = np.zeros(self.nbf * self.actions.shape[0])

            if e != 0.0:
                # add new sample (e-greedy)
                a = None
                if np.random.rand() <= e or j < update_rate:
                    # take random action
                    a = self.env.action_space.sample()
                else:
                    # take best action under current value function
                    a = self.get_action(s)

                if e > e_min:
                    e *= e_decay
                else:
                    e = 0.0


            sprime, r, done, info = self.env.step(a)
            samples.append(np.concatenate((s, a, sprime, [r])))
            if done:
                s = self.env.reset()
            else:
                s = sprime

            if j % update_rate == 0 or e == 0.0:
                # update the value function (run full LSTDQ iteration)
                print("Starting LSTDQ run #" + str(i) + ", epsilon = " + str(e))
                for sample in samples:
                    # samples: s, a, s', r
                    state = sample[0:self.state_dim]
                    action = sample[self.state_dim]
                    next_state = sample[self.state_dim + self.action_dim:self.state_dim * 2 + self.action_dim]
                    reward = sample[-1]

                    phi = self.basis_functions.transform(state, action).reshape(-1)
                    best_action = self.get_action(next_state)
                    phi_prime = self.basis_functions.transform(next_state, best_action).reshape(-1)

                    A = A + np.outer(phi, (phi - gamma * phi_prime))
                    b = b + phi * reward

                last_w = np.copy(self.w)

                if np.linalg.matrix_rank(A) == A.shape[0]:
                    self.w = np.linalg.solve(A, b)
                else:
                    self.w = lstsq(A, b)[0]

#                    self.w = lsqr(A, b, damp=self.damp)[0]

                threshold = np.linalg.norm(self.w - last_w)
                print("Norm ||w - w'||:", threshold)
                print("Weights:", self.w)
                print()

                i = i + 1

            j = j + 1

        print("Done with LSPI iteration, final weights:")
        print(self.w)
        print()


    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            lspi = pickle.load(f)

        return lspi
