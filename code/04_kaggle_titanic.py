'''

Classification and Regression Trees

'''
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

'''

Read, Explore, and Process data

'''

# Read in the data
titanic = pd.read_csv('../data/titanic.csv')

titanic.head()

titanic.describe()


# Take a  selection of the variables
d = titanic[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

# Check for missing values in all columns
d.isnull().sum()
d.groupby(['Sex', 'Pclass']).Age.apply(lambda x: x.isnull().sum()) / d.groupby(['Sex', 'Pclass']).Age.count()

# Convert all variables to numeric so for scikit learn
d['Sex'] = np.where(d.Sex == 'female', 1, 0)

# Fill in missing values with the mean value (hint: use .fillna())
d['Age'] = d['Age'].fillna(d['Age'].mean())

# Explore the data to indtify trends in characteristics of survivors
d.Survived.value_counts()                    # How many people lived and died
d.Survived.mean()                            # The survival rate for everyone
d.groupby('Sex').Survived.mean()             # By Sex: women have higher survival rates
d.groupby('Pclass').Survived.mean()          # By Pclass: 1st class passengers have higher survival rates
d.groupby(['Sex', 'Pclass']).Survived.mean() # By Sex and Pclass: Women in the 1st and 2nd classes had the highest survival rates

# Create a proxy variable representing whether the Spouse was on board
d['Spouse'] = ((d.Age > 18) & (d.SibSp >= 1)).astype(int)
d.Spouse.value_counts()
d.groupby(['Pclass', 'Spouse']).Survived.mean() # Having a spouse appears to increase survival in the 1st class only


'''
Split into training and test datasets, and build the model
'''

# Now, split the data into training and test sets
train, test = train_test_split(d,test_size=0.3, random_state=1)

# Convert them back into dataframes, for convenience
train = pd.DataFrame(data=train, columns=d.columns)
test = pd.DataFrame(data=test, columns=d.columns)

# Create a decision tree classifier instance (start out with a small tree for interpretability)
ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

feature_df = train.drop('Survived', axis=1)
survived = train['Survived']

# Fit the decision tree classider
ctree.fit(feature_df, survived)


# How to interpret the tree?
ctree.classes_

# Create a feature vector
features = d.columns.tolist()[1:]

# Predict what will happen for 1st class woman
features
ctree.predict_proba([1, 1, 25, 0, 0, 0])
ctree.predict([1, 1, 25, 0, 0, 0])

# Predict what will happen for a 3rd class man
ctree.predict_proba([3, 0, 25, 0, 0, 0])

# How about a woman?
ctree.predict_proba([3, 1, 25, 0, 0, 0])


# Which features are the most important?
ctree.feature_importances_

# Clean up the output
pd.DataFrame(zip(features, ctree.feature_importances_)).sort_index(by=1, ascending=False)

# Make predictions on the test set
preds = ctree.predict(test.drop('Survived', axis=1))

# Calculate accuracy
metrics.accuracy_score(test['Survived'], preds)

# Confusion matrix
pd.crosstab(test['Survived'], preds, rownames=['actual'], colnames=['predicted'])

# Make predictions on the test set using predict_proba
probs = ctree.predict_proba(test.drop('Survived', axis=1))[:,1]

# Calculate the AUC metric
metrics.roc_auc_score(test['Survived'], probs)

'''

FINE-TUNING THE TREE

'''

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

y = d['Survived'].values
X = d[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Spouse']].values

# check CV score for max depth = 3
ctree = tree.DecisionTreeClassifier(max_depth=3)
np.mean(cross_val_score(ctree, X, y, cv=5, scoring='roc_auc'))

# check CV score for max depth = 10
ctree = tree.DecisionTreeClassifier(max_depth=10)
np.mean(cross_val_score(ctree, X, y, cv=5, scoring='roc_auc'))

# Conduct a grid search for the best tree depth
ctree = tree.DecisionTreeClassifier(random_state=1, min_samples_leaf=20)
depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(X, y)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

grid_mean_scores


# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_

best

# Read in test data from site
test = pd.read_csv('../data/titanic_test.csv')

# Do all of the same transformations we did above to this set
test['Sex'] = np.where(test.Sex == 'female', 1, 0)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Spouse'] = ((test.Age > 18) & (test.SibSp >= 1)).astype(int)

# predict our out of sample data using our "Best" model
predictions = best.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Spouse']].values)

submission_df = pd.DataFrame(test['PassengerId'])
submission_df['Survived'] = predictions

# Make our submission
submission_df.to_csv('submission.csv', index = False)


# Congratualtions, you are now a ranking data scientist :)