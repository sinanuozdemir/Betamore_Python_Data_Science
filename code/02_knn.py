'''
CLASS: Introduction to scikit-learn with iris data
'''

import numpy as np
import pandas as pd
# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()

# create X (features) and y (response)
X, y = iris.data, iris.target
X.shape
y.shape
iris.feature_names
iris.target_names

# Convert data to a pandas data frame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['Species'] = iris.target

# predict y with KNN
from sklearn.neighbors import KNeighborsClassifier    # import class
knn = KNeighborsClassifier(n_neighbors=1)             # instantiate the estimator
knn.fit(iris_df[iris.feature_names], iris_df.Species) # fit with data
knn.predict([3, 5, 4, 2])                             # predict for a new observation
iris.target_names[knn.predict([3, 5, 4, 2])]
knn.predict([3, 5, 2, 2])

# predict for multiple observations at once
X_new = [[3, 5, 4, 2], [3, 5, 2, 2]]
knn.predict(X_new)

# try a different value of K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(iris_df[iris.feature_names], iris_df.Species)
knn.predict(X_new)              # predictions
knn.predict_proba(X_new)        # predicted probabilities
knn.kneighbors([3, 5, 4, 2])    # distances to nearest neighbors (and identities)
np.sqrt(((X[106] - [3, 5, 4, 2])**2).sum()) # Euclidian distance calculation for nearest neighbor

# compute the accuracy for K=5 and K=1
# accuracy = percentage of correct predictions
# Note hsre I'm passing in numpy arrays into my classifier instead of the dataframe
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.score(X, y)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)

