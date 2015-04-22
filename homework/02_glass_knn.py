'''
HOMEWORK SOLUTION: Glass Identification
'''

import pandas as pd
import numpy as np


## PART 1

# read data into a DataFrame
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',
                 header=None, names=['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type'],
                 index_col='id')

# briefly explore the data
df.head()
df.tail()
df.isnull().sum()

# convert to binary classification problem:
#   types 1/2/3/4 are mapped to 0
#   types 5/6/7 are mapped to 1
df['binary'] = np.where(df.glass_type < 5, 0, 1)


## PART 2

# create a list of features (make sure not to use 'id' or 'glass_type' as features!)
feature_cols = ['ri','na','mg','al','si','k','ca','ba','fe']

# define X (features) and y (response)
X = df[feature_cols]        # features
y = df.binary               # binary response
y2 = df.glass_type          # multinomial response

## PART 3

# fit a KNN model and make predictions with the binary response. Use 1 and 3 neighbors
# Score each model using the .score method






# fit a KNN model and make predictions with the binary response. Use 1 and 3 neighbors
# Score each model using the .score method




# What is the null accuracy rate of the binary data?


