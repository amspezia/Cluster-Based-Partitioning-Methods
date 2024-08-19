import numpy as np
from sklearn.utils import indexable, check_random_state, shuffle
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
import copy

from .utils import cluster_labels_to_folds


def DBSCANBCV(X, y, k_splits, eps, min_samples):    
    if k_splits == None:
        k_splits = len(np.unique(y)) if y is not None else 2
    np.set_printoptions(threshold=np.inf)

    dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
    cluster_labels = dbscan.fit_predict(X) 
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    folds = cluster_labels_to_folds(cluster_labels, k_splits)

    return folds, k_splits, n_clusters


class DBSCANBCVSplitter:
    def __init__(self, n_splits=None, n_clusters=None, random_state=None, shuffle=True, eps=0.5, min_samples=5):
        """Split dataset indices according to the DBSCAN technique.

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
        self.eps = eps
        self.min_samples = min_samples

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

        folds, self.n_splits, self.n_clusters = DBSCANBCV(
            X, y,self.n_splits, self.eps, self.min_samples)
        
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