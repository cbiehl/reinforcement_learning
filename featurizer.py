import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import RBFSampler
from scipy.spatial.distance import cdist, pdist


class ApproximateRBFGenerator(BaseEstimator, TransformerMixin):
    """
    Generates Random Fourier Features to approximate an RBF Kernel
    """

    def __init__(self, action_bins, n_basisfunctions=10, normalize=False):
        self.nbf = n_basisfunctions
        self.action_bins = action_bins
        self.normalize = normalize
        self.scaler = StandardScaler()


    def get_action_index(self, action):
        """
        Discretizes an action (scalar) according to bins in self.actions
        and returns the corresponding index in self.actions

        :param action:
        :return:
        """
        idx = np.searchsorted(self.action_bins, action).astype(int) - 1

        if idx < 0:
            idx = 0
        elif idx > self.action_bins.shape[0] - 2:
            idx = self.action_bins.shape[0] - 2

        return idx


    def fit(self, X, y=None):
        X = np.asarray(X)

        if self.normalize:
            X = self.scaler.fit_transform(X)

        # choose nu according to:
        # http://papers.nips.cc/paper/7233-towards-generalization-and-simplicity-in-continuous-control.pdf
        distances = pdist(X, metric='euclidean')
        nu = np.sum(distances) / distances.shape[0]
        self.rbfsampler = RBFSampler(nu, self.nbf - 1)
        self.rbfsampler.fit(X)
        print("Fitted Random Fourier Features using nu = " + str(nu))


    def transform(self, X, action):
        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.asarray([X])

        if self.normalize:
            X = self.scaler.transform(X)

        if type(action) in (np.ndarray, list) and np.asarray(action).shape[0] < 2:
            action = action[0]

        state_features = np.concatenate((np.asarray([1.0]), self.rbfsampler.transform(X)[0]))
        features = np.zeros(self.nbf * (self.action_bins.shape[0] - 1))
        offset = self.get_action_index(action) * state_features.shape[0]
        features[offset:offset + state_features.shape[0]] = state_features

        return features


class FourierFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, action_bins, n_basisfunctions, nu='auto', normalize=True):
        self.nu = nu
        self.nbf = n_basisfunctions
        self.action_bins = action_bins
        self.normalize = normalize
        self.scaler = StandardScaler()


    def get_action_index(self, action):
        """
        Discretizes an action (scalar) according to bins in self.actions
        and returns the corresponding index in self.actions

        :param action:
        :return:
        """
        idx = np.searchsorted(self.action_bins, action).astype(int) - 1

        if idx < 0:
            idx = 0
        elif idx > self.action_bins.shape[0] - 2:
            idx = self.action_bins.shape[0] - 2

        return idx


    def fit(self, X, y=None):
        if self.normalize:
            X = self.scaler.fit_transform(X)

        if self.nu == 'auto':
            # choose nu according to:
            # http://papers.nips.cc/paper/7233-towards-generalization-and-simplicity-in-continuous-control.pdf
            distances = pdist(X, metric='euclidean')
            self.nu = np.sum(distances) / distances.shape[0]

        print("Fitted Random Fourier Features using nu = " + str(self.nu))


    def transform(self, X, action):
        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.asarray([X])

        if self.normalize:
            X = self.scaler.transform(X.reshape(1, -1))

        if type(action) in (np.ndarray, list) and np.asarray(action).shape[0] < 2:
            action = action[0]

        state_features = np.sin((np.sum(np.random.normal(0.0, 1.0, size=(self.nbf - 1, X.shape[0])) * X, axis=1) / self.nu)\
                                + np.random.uniform(low=-np.pi, high=np.pi, size=self.nbf - 1))

        state_features = np.concatenate((np.asarray([1.0]), state_features))
        features = np.zeros(self.nbf * self.action_bins.shape[0])
        offset = self.get_action_index(action) * state_features.shape[0]
        features[offset:offset + state_features.shape[0]] = state_features

        return features


class RBFFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Class feature_generator.
    """

    def __init__(self, action_bins, n_basisfunctions=10, normalize=False):
        """
        Constructor.

        :param xmin:
        :param xmax:
        """
        self.nbf = n_basisfunctions
        self.action_bins = action_bins
        self.normalize = normalize
        self.scaler = StandardScaler()


    def get_action_index(self, action):
        """
        Discretizes an action (scalar) according to bins in self.actions
        and returns the corresponding index in self.actions

        :param action:
        :return:
        """
        idx = np.searchsorted(self.action_bins, action).astype(int) - 1

        if idx < 0:
            idx = 0
        elif idx > self.action_bins.shape[0] - 2:
            idx = self.action_bins.shape[0] - 2

        return idx


    def fit(self, X, y=None):
        X = np.asarray(X)

        if self.normalize:
            X = self.scaler.fit_transform(X)

        k_means = KMeans(n_clusters=self.nbf - 1).fit(X)

        self.means = k_means.cluster_centers_
        self.stds = []

        d = { i: X[np.where(k_means.labels_ == i)[0]] for i in range(k_means.n_clusters) }
        for key, val in d.items():
            self.stds.append(np.std(val))

        self.stds = np.asarray(self.stds)


    def transform(self, X, action):
        X = np.asarray(X)
        if len(X.shape) < 2:
            X = np.asarray([X])

        if self.normalize:
            X = self.scaler.transform(X.reshape(1, -1))

        if type(action) in (np.ndarray, list) and np.asarray(action).shape[0] < 2:
            action = action[0]

        pairwise_distance = cdist(X, self.means, metric='sqeuclidean')
        rbf_dist = np.exp(-pairwise_distance / (2 * self.stds**2))

        state_features = np.concatenate((np.asarray([1.0]), rbf_dist.reshape(-1)), axis=0)
        features = np.zeros(self.nbf * (self.action_bins.shape[0] - 1))
        offset = self.get_action_index(action) * state_features.shape[0]
        features[offset:offset + state_features.shape[0]] = state_features

        return features
