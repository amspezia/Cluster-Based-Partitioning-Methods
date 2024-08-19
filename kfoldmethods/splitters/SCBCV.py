import numpy as np
from sklearn.utils import indexable, check_random_state, shuffle
from sklearn.cluster import KMeans, MiniBatchKMeans
import copy

from .utils import aux_indexes, circular_append, sort_by_label, data_slicing_by_label


def SCBCV(X, y, k_splits, k_clusters, rng=None, minibatch_kmeans=False):
    if rng is None:
        rng = np.random.RandomState()
    
    if k_splits == None and k_clusters == None:
        k_splits = len(np.unique(y))  # extrating k, the k that will be used on clustering, from y
        k_clusters = k_splits

    if k_clusters == None:
        k_clusters = k_splits

    X, y = sort_by_label(X, y) 
    slicing_index, segment_shift, start_of_segment, end_of_segment = aux_indexes(y)
    X, y = data_slicing_by_label(X, y, slicing_index)

    index_list = []

    if minibatch_kmeans:
        kmeans = MiniBatchKMeans(n_clusters=k_clusters, random_state=rng)
    else:
        kmeans = KMeans(n_clusters=k_clusters, random_state=rng)
    
    for each_class in X: 
        X_new = kmeans.fit_transform(each_class) 
        cluster_index = np.argsort(X_new)
        cluster_index = [i[0] for i in cluster_index]
        
        clusters = [[] for _ in range(k_clusters)]  # list with k clusters (empty)
        for i in range(len(X_new)):
            element = (i, X_new[i][cluster_index[i]])
            clusters[cluster_index[i]].append(element)

        dtype = [('index', int), ('distance', float)]
        for values in clusters:
            each_cluster = np.array(values, dtype=dtype)
            each_cluster = np.sort(each_cluster, order='distance')
            index_list.extend(each_cluster['index'])

    for i, j, x in zip(start_of_segment, end_of_segment, segment_shift):
        index_list[i:j] += x
        
    folds = [[] for _ in range(k_splits)]
    folds = circular_append(index_list, folds, k_splits)

    return folds, k_splits, k_clusters


class SCBCVSplitter:
    def __init__(self, n_splits=None, n_clusters = None, random_state=None, shuffle=True, 
                 minibatch_kmeans=False):
        """Split dataset indices according to the SCBCV technique.

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
        self.minibatch_kmeans = minibatch_kmeans

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

        folds, self.n_splits, self.n_clusters = SCBCV(
            X, y,self.n_splits, self.n_clusters, rng=rng, minibatch_kmeans=self.minibatch_kmeans)
        
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

