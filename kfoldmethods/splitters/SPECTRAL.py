import numpy as np
from sklearn.utils import indexable, check_random_state, shuffle
from sklearn.cluster import SpectralClustering
import copy

from .utils import circular_append


def SPECTRAL(X, y, k_splits, k_clusters, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    
    if k_splits == None and k_clusters == None:
        k_splits = len(np.unique(y))  # extrating k, the k that will be used on clustering, from y
        k_clusters = k_splits

    if k_clusters == None:
        k_clusters = k_splits

    spectral = SpectralClustering(n_clusters=k_clusters, random_state=rng)
    cluster_labels = spectral.fit_predict(X)  # does not allow to choose the metric for distance
    
    clusters = [[] for _ in range(k_clusters)]  # list with k clusters (empty)
    for i in range(len(cluster_labels)):
        clusters[cluster_labels[i]].append(i)
    
    index_list = []
    for list in clusters:
        index_list.extend(list)

    folds = [[] for _ in range(k_splits)]
    folds = circular_append(index_list, folds, k_splits)

    return folds, k_splits, k_clusters


class SPECTRALSplitter:
    def __init__(self, n_splits=None, n_clusters = None, random_state=None, shuffle=True):
        """Split dataset indices according to the SPECTRAL technique.

        Parameters
        ----------
        n_splits : int
            Number of splits to generate. In this case, this is the same as the K in a K-fold cross validation.
        random_state : any
            Seed or numpy RandomState. If None, use the singleton RandomState used by numpy.
        shuffle : bool
            Shuffle dataset before splitting.
        """
        # in sklearn, generally, we do not check the arguments in the initialization function.
        # There is a reason for this.
        self.n_splits = n_splits
        self.n_clusters = n_clusters
        self.random_state = random_state  # used for enabling the user to reproduce the results
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        """Generate indices to split data according to the DBSCV technique.

        Parameters
        ----------
        X : array-like object of shape (n_samples, n_features)
            Training data.
        y : array-like object of shape (n_samples, )
            Target variable corresponding to the training data.
        groups : None
            Not implemented. It is here for compatibility.

        Yields
        -------
            Split with train and test indexes.
        """
        if groups:
            raise NotImplementedError("groups functionality is not implemented.")

        # just some validations that sklearn uses
        X, y = indexable(X, y)
        rng = check_random_state(self.random_state)

        if self.shuffle:
            X, y = shuffle(X, y, random_state=rng)

        folds, self.n_splits, self.n_clusters = SPECTRAL(
            X, y,self.n_splits, self.n_clusters, rng=rng)
        
        for k in range(self.n_splits):
            test_fold_index = self.n_splits - k - 1  # start by using the last fold as the test fold
            ind_train = []
            ind_test = []
            for fold_index in range(self.n_splits):
                if fold_index != test_fold_index:
                    ind_train += copy.copy(folds[fold_index])
                else:
                    ind_test = copy.copy(folds[fold_index])

            yield ind_train, ind_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits