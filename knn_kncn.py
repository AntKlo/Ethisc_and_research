from sklearn.neighbors import NearestNeighbors, NearestCentroid
import numpy as np
from collections import Counter
from copy import deepcopy


def depuration(X_train, y_train, k, k_prim):
    """Depuration function examines training set and for every item in it
    it checks if the item has at least k_prim neighbors with the label
    the same as the original label of this item. If there is no enough
    such neighbors, discard this the item from training set.

    Args:
        X_train (array): training set
        y_train (array): labels for training set
        k (int): number of neighbors
        k_prim (int): number of minimum neighbors with the same label

    Raises:
        ValueError: k should be integer greater than 0
        ValueError: k_prim should be in range [(k+1)/2, k]

    Returns:
        two lists: list with edited training set and list with labels
    """
    if not isinstance(k, int):
        raise TypeError('k should be an integer greater than 0')
    if k < 1:
        raise ValueError('k should be an integer greater than 0')
    if not isinstance(k_prim, int):
        raise TypeError('k_prim should be an integer greater than 0')
    if k_prim > k or k_prim < (k+1)/2:
        raise ValueError('k_prim should be in range [(k+1)/2, k]')
    Sx = []
    Sy = []
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train)
    for x, y in zip(X_train, y_train):
        neighbors = knn.kneighbors([x], k+1, return_distance=False)[0][1:]
        labels = Counter([y_train[neigh] for neigh in neighbors])
        for key in labels:
            if labels[key] >= k_prim:
                Sx.append(x)
                Sy.append(y)

    return Sx, Sy


def kncn_edit(X_train, y_train):
    """kncn_edit function examines training set and for every item in it
    it checks if the prediction for this item, using k nearest centroids method
    (centroid space created with the rest items in the training set),
    is the same as the original label o this item.

    Args:
        X_train (array): training set
        y_train (array): labels for training set

    Returns:
        two lists: list with edited training set and list with labels
    """
    n = len(y_train)
    Sx = []
    Sy = []
    # Calculations for the first element
    knc = NearestCentroid()
    X_tr = X_train[1:]
    y_tr = y_train[1:]
    knc.fit(X_tr, y_tr)
    label = knc.predict([X_train[0]])
    if label == y_train[0]:
        Sx.append(X_train[0])
        Sy.append(y_train[0])
    # Calculations for the 2nd element to the one before last
    for i in range(1, n-1):
        X_tr = np.concatenate((X_train[0:i], X_train[i+1:]))
        y_tr = np.concatenate((y_train[0:i], y_train[i+1:]))
        knc = NearestCentroid()
        knc.fit(X_tr, y_tr)
        label = knc.predict([X_train[i]])
        if label == y_train[i]:
            Sx.append(X_train[i])
            Sy.append(y_train[i])
    # Calculations for the last element
    knc = NearestCentroid()
    X_tr = X_train[:n-1]
    y_tr = y_train[:n-1]
    knc.fit(X_tr, y_tr)
    label = knc.predict([X_train[n-1]])
    if label == y_train[n-1]:
        Sx.append(X_train[n-1])
        Sy.append(y_train[n-1])

    return Sx, Sy


def iterative_kncn_edit(X_train, y_train):
    """Iterative kncn_edit function works in the same way as
    the kncn_edit function but in the while loop. It processes
    the training set until there is no item missclasified
    (the training set becomes smaller each iteration).

    Args:
        X_train (array): training set
        y_train (array): labels for training set

    Returns:
        two lists: list with edited training set and list with labels
    """
    X_train_new, y_train_new = kncn_edit(X_train, y_train)
    while len(X_train_new) != len(X_train):
        X_train = deepcopy(X_train_new)
        y_train = deepcopy(y_train_new)
        X_train_new, y_train_new = kncn_edit(X_train, y_train)

    return X_train_new, y_train_new