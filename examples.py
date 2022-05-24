from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import KFold
from knn_kncn import *

# Load dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target


# --------DEPURATION EXAMPLE--------

# cross-validation depuration
for k in range(5, 10, 2):
    scores = []
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        scores.append(score)
    print(
        f'Mean CV accuracy for standard knn classifier with {k} neighbors: {sum(scores)/len(scores)}')

for k in range(5, 10, 2):
    scores = []
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = depuration(X_train, y_train, k, k-1)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        scores.append(score)
    print(
        f'Mean CV accuracy for knn classifier with depuration with {k} neighbors: {sum(scores)/len(scores)}')


# --------KNC_EDIT EXAMPLE--------

# cross-validation kncn_edit
scores = []
knn_classifier = KNeighborsClassifier(n_neighbors=k)
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scores.append(score)
print(
    f'Mean CV accuracy for standard knc classifier: {sum(scores)/len(scores)}')


scores = []
knn_classifier = KNeighborsClassifier(n_neighbors=k)
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, y_train = kncn_edit(X_train, y_train)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scores.append(score)
print(
    f'Mean CV accuracy for edited knc classifier: {sum(scores)/len(scores)}')



# --------ITERATIVE KNC_EDIT EXAMPLE--------

# cross-validation iterative_kncn_edit
scores = []
knn_classifier = KNeighborsClassifier(n_neighbors=k)
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scores.append(score)
print(
    f'Mean CV accuracy for standard knc classifier: {sum(scores)/len(scores)}')

scores = []
knn_classifier = KNeighborsClassifier(n_neighbors=k)
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, y_train = iterative_kncn_edit(X_train, y_train)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scores.append(score)
print(
    f'Mean CV accuracy for iteratively edited knc classifier: {sum(scores)/len(scores)}')