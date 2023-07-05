import numpy as np

import torch

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from gbngboost import GeneBayesNGBRegressor
from prior import Prior, posterior_summary

import matplotlib.pyplot as plt


#----------------------------------------------------------
# Simulation
#----------------------------------------------------------

np.random.seed(42)

PARAMETERS = 200
OBSERVATIONS = 20000

# Create feature matrix X
X = np.random.randn(OBSERVATIONS, PARAMETERS)

# Create unobserved variable U
# The mean and variance of U are a function of X
U = (np.random.randn(OBSERVATIONS) + X @ np.random.randn(PARAMETERS) / PARAMETERS) * np.exp((X @ np.random.randn(PARAMETERS)) / PARAMETERS)

# Create the observed variances F
F = np.exp(np.random.randn(OBSERVATIONS)).reshape(-1, 1)

# Create the observed variable Y
Y = (np.random.randn(OBSERVATIONS) + U) * np.sqrt(F.flatten())

# Split into training and testing
X_train, X_test, Y_train, Y_test, F_train, F_test, U_train, U_test = train_test_split(X, Y, F, U, test_size=0.1)

# Plot Y against U
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(U_train, Y_train, s=10, c='black', alpha=0.5, label='Train')
ax.scatter(U_test, Y_test, s=10, c='firebrick', alpha=0.5, label='Test')
ax.set_xlabel('U')
ax.set_ylabel('Y')
ax.set_title('Simulated Training and Testing Data')
ax.legend()

#----------------------------------------------------------
# Training
#----------------------------------------------------------

if not torch.cuda.is_available():
    learner = XGBRegressor(tree_method='hist')
else:
    learner = XGBRegressor(gpu_id=0, tree_method='gpu_hist')

ngb = GeneBayesNGBRegressor(
    Dist=Prior, Base=learner,
    learning_rate=0.001, early_stopping_rounds=10,
    n_estimators=500,
).fit(X_train, Y_train, F_train)
Y_preds = ngb.predict(X_test, F_test)
Y_dists = ngb.pred_dist(X_test, F_test)

#----------------------------------------------------------
# Test Set Performance for Predicting Y
#----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Y_test, Y_preds)
ax.set_xlabel('Test Y')
ax.set_ylabel('Predicted Y')

#----------------------------------------------------------
# MAP for U
#----------------------------------------------------------

prior_mean, post_mean, post_lower, post_upper = posterior_summary(X_train, Y_train, F_train, ngb)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(U_train, post_mean)
ax.set_xlabel('Train U')
ax.set_ylabel('Posterior U MAP')

prior_mean, post_mean, post_lower, post_upper = posterior_summary(X_test, Y_test, F_test, ngb)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(U_test, post_mean)
ax.set_xlabel('Test U')
ax.set_ylabel('Posterior U MAP')

