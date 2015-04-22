'''
Linear Regression
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''
Exploring the Data
'''

# Dataset: Sales and advertising channels
adv = pd.read_csv('../data/Advertising.csv')
adv.head()


# Plot the Radio spending against the Sales
plt.figure(figsize=(10,9))
plt.subplot(221)
plt.scatter(adv.Radio, adv.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("Radio"); plt.ylabel("Sales")

# Plot the Newspaper spending against the Sales
plt.subplot(222)
plt.scatter(adv.Newspaper, adv.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("Newspaper"); plt.ylabel("Sales")

# Plot the TV spending against the Sales
plt.subplot(223)
plt.scatter(adv.TV, adv.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("TV"); plt.ylabel("Sales")

# Plot the Region against the Sales
plt.subplot(224)
plt.scatter(adv.Region, adv.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("Region"); plt.ylabel("Sales")

'''
ESTIMATING THE COEFFICIENTS
'''

# Fit a linear regression model using OLS
from sklearn.linear_model import LinearRegression
slm = LinearRegression()
slm.fit(adv['Radio'][:,np.newaxis], adv['Sales'])

# Evaluate the output
slm.intercept_
slm.coef_

preds = slm.predict(adv['Radio'][:,np.newaxis])  # list of predictions


# Plot our points against our line

# Calculate the ymin, ymax
ymin = slm.predict(adv.Radio.min())
ymax = slm.predict(adv.Radio.max())

# Plotting
plt.plot([adv.Radio.min(), adv.Radio.max()], [ymin, ymax])
plt.scatter(adv.Radio, adv.Sales)


# Working with multiple features
slm = LinearRegression()
slm.fit(adv[['Radio', 'TV']], adv['Sales'])

# Evaluate the output
slm.intercept_
slm.coef_

preds = slm.predict(adv[['Radio', 'TV']])


# How can we evaluate our linear regression model??
# Hard to graph multi-feature linear models..

# HOMEWORK

# Create a linear regression to predict Sales based on all variables
# "choose" which models you think are the best
# Next time in class we will put them to the test!