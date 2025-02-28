# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
#

# %% [markdown]
# # **Regression Trees**
#

# %% [markdown]
# Estimated time needed: **30** minutes
#

# %% [markdown]
# In this exercise session you will use a real dataset to train a regression tree model. The dataset includes information about taxi tip and was collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). You will use the trained model to predict the amount of tip paid.
#
#

# %% [markdown]
# ## Objectives
#

# %% [markdown]
# After completing this lab you will be able to:
#

# %% [markdown]
# * Perform basic data preprocessing using Scikit-Learn
# * Model a regression task using Scikit-Learn
# * Train a Decision Tree Regressor model
# * Run inference and assess the quality of the trained models
#

# %% [markdown]
# <h2>Introduction</h2>
# The dataset used in this exercise session is a subset of the publicly available <a><href='https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page'>TLC Dataset</a> (all rights reserved by Taxi & Limousine Commission (TLC), City of New York). The prediction of the tip amount can be modeled as a regression problem. To train the model you can use part of the input dataset and the remaining data can be used to assess the quality of the trained model.
#     <br>
# </div>
#

# %% [markdown]
# <div id="import_libraries">
#     <h2>Import Libraries</h2>
# </div>
#

# %% [markdown]
# Make sure the libraries required are available by executing the cell below.
#

# %%
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.tree import DecisionTreeRegressor
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn

# %% [markdown]
# Import the libraries we need to use in this lab
#

# %%
%matplotlib inline


warnings.filterwarnings('ignore')

# %% [markdown]
# <div id="dataset_analysis">
#     <h2>Dataset Analysis</h2>
# </div>
#

# %% [markdown]
# In this section you will read the dataset in a Pandas dataframe and visualize its content. You will also look at some data statistics.
#
# Note: A Pandas dataframe is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure. For more information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html.
#

# %%
# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data

# %% [markdown]
# Each row in the dataset represents a taxi trip. As shown above, each row has 13 variables. One of the variables is `tip_amount` which will be the target variable. Your objective will be to train a model that uses the other variables to predict the value of the `tip_amount` variable.
#

# %% [markdown]
# To understand the dataset a little better, let us plot the correlation of the target variable against the input variables.
#

# %%
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# %% [markdown]
# This shows us that the input features `payment_type`, `VendorID`, `store_and_fwd_flag` and `improvement_surcharge` have little to no correlation with the target variable.
#

# %% [markdown]
# <div id="dataset_preprocessing">
#     <h2>Dataset Preprocessing</h2>
# </div>
#

# %% [markdown]
# You will now prepare the data for training by applying normalization to the input features.
#

# %%
# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

# %% [markdown]
# <div id="dataset_split">
#     <h2>Dataset Train/Test Split</h2>
# </div>
#

# %% [markdown]
# Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
#

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %% [markdown]
# <div id="dt_sklearn">
#     <h2>Build a Decision Tree Regressor model with Scikit-Learn</h2>
# </div>
#

# %% [markdown]
# Regression Trees are implemented using `DecisionTreeRegressor`.
#
# The important parameters of the model are:
#
# `criterion`: The function used to measure error, we use 'squared_error'.
#
# `max_depth` - The maximum depth the tree is allowed to take; we use 8.
#

# %%
# import the Decision Tree Regression Model from scikit-learn

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion='squared_error',
                               max_depth=8,
                               random_state=35)

# %% [markdown]
# Now lets train our model using the `fit` method on the `DecisionTreeRegressor` object providing our training data
#

# %%
dt_reg.fit(X_train, y_train)

# %% [markdown]
# <div id="dt_sklearn_snapml">
#     <h2>Evaluate the Scikit-Learn and Snap ML Decision Tree Regressor Models</h2>
# </div>
#

# %% [markdown]
# To evaluate our dataset we will use the `score` method of the `DecisionTreeRegressor` object providing our testing data, this number is the $R^2$ value which indicates the coefficient of determination. We will also evaluate the Mean Squared Error $(MSE)$ of the regression output with respect to the test set target values. High $R^2$ and low $MSE$ values are expected from a good regression model.
#

# %%
# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

# %% [markdown]
# ## Practice
#

# %% [markdown]
# Q1. What if we change the max_depth to 12? How would the $MSE$ and $R^2$ be affected?
#

# %%
dt_reg_q1 = DecisionTreeRegressor(
    criterion='squared_error', max_depth=12, random_state=35)
dt_reg_q1.fit(X_train, y_train)

y_pred_q1 = dt_reg_q1.predict(X_test)

# MSE
mse_q1 = mean_squared_error(y_test, y_pred_q1)
print('MSE score : {0:.3f}'.format(mse_q1))

# R2
r2_score_q1 = dt_reg_q1.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score_q1))

# %% [markdown]
# <details><summary>Click here for the solution</summary>
# MSE is noted to be increased by increasing the max_depth of the tree. This may be because of the model having excessive parameters due to which it overfits to the training data, making the performance on the testing data poorer. Another important observation would be that the model gives a <b>negative</b> value of $R^2$. This again indicates that the prediction model created does a very poor job of predicting the values on a test set.
# </details>
#

# %% [markdown]
# Q2. Identify the top 3 features with the most effect on the `tip_amount`.
#

# %%
# your code here
corr_values = raw_data.corr()['tip_amount']
print(abs(corr_values).sort_values(ascending=False))
corr_values.plot(kind='barh', figsize=(10, 6))

# %%
three_top_features = ['fare_amount', 'tolls_amount', 'trip_distance']

# %% [markdown]
# <details><summary>Click here for the solution</summary>
#
# ```python
# correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
# abs(correlation_values).sort_values(ascending=False)[:3]
#
# ```
# <br>
# As is evident from the output, Fare amount, toll amount and trip distance are the top features affecting the tip amount, which make logical sense.
# </details>
#

# %% [markdown]
# Q3. Since we identified 4 features which are not correlated with the target variable, try removing these variables from the input set and see the effect on the $MSE$ and $R^2$ value.
#

# %%
dataq3 = raw_data[three_top_features + ['tip_amount']]
Xq3 = dataq3.drop(columns='tip_amount')
yq3 = dataq3[['tip_amount']]

X_train_q3, X_test_q3, y_train_q3, y_test_q3 = train_test_split(
    Xq3, yq3, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train_q3 = sc.fit_transform(X_train_q3)
X_test_q3 = sc.transform(X_test_q3)

dt_reg_q3 = DecisionTreeRegressor(
    criterion='squared_error', max_depth=8, random_state=35)
dt_reg_q3.fit(X_train_q3, y_train_q3)

y_pred_q3 = dt_reg_q3.predict(X_test_q3)

# MSE
mse_q3 = mean_squared_error(y_test_q3, y_pred_q3)
print('MSE score : {0:.3f}'.format(mse_q1))

# R2
r2_score_q3 = dt_reg_q3.score(X_test_q3, y_test_q3)
print('R^2 score : {0:.3f}'.format(r2_score_q3))

# %% [markdown]
# <details><summary>Click here for the solution</summary>
#
# ```python
# raw_data = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
#
# # Execute all the cells of the lab after modifying the raw data.
# ```
# <br>
# The MSE and $R^2$ values does not change significantly, showing that there is minimal affect of these parameters on the final regression output.
# </details>
#

# %% [markdown]
# Q4. Check the effect of **decreasing** the `max_depth` parameter to 4 on the $MSE$ and $R^2$ values.
#

# %%
dt_reg_q4 = DecisionTreeRegressor(
    criterion='squared_error', max_depth=4, random_state=35)
dt_reg_q4.fit(X_train_q3, y_train_q3)

y_pred_q4 = dt_reg_q4.predict(X_test_q3)

# MSE
mse_q4 = mean_squared_error(y_test_q3, y_pred_q4)
print('MSE score : {0:.3f}'.format(mse_q4))

# R2
r2_score_q4 = dt_reg_q4.score(X_test_q3, y_test_q3)
print('R^2 score : {0:.3f}'.format(r2_score_q4))

# %% [markdown]
# <details><summary>Click here for the solution</summary>
# You will note that the MSE value decreases and $R^2$ value increases, meaning that the choice of `max_depth=4` may be more suited for this dataset.
# </details>
#

# %% [markdown]
# ### Congratulations! You're ready to move on to your next lesson!
#
# ## Author
# <a href="https://www.linkedin.com/in/abhishek-gagneja-23051987/" target="_blank">Abhishek Gagneja</a>
#
# ### Other Contributors
# <a href="https://www.linkedin.com/in/jpgrossman/" target="_blank">Jeff Grossman</a>
#
# <h3 align="center"> © IBM Corporation. All rights reserved. <h3/>
#
#
# <!--
# ## Change Log
#
#
# |  Date (YYYY-MM-DD) |  Version       | Changed By     | Change Description                  |
# |---|---|---|---|
# | 2024-10-31         | 3.0            | Abhishek Gagneja  | Rewrite                             |
# | 2020-11-03         | 2.1            | Lakshmi        | Made changes in URL                 |
# | 2020-11-03         | 2.1            | Lakshmi        | Made changes in URL                 |
# | 2020-08-27         | 2.0            | Lavanya        | Moved lab to course repo in GitLab  |
# |   |   |   |   |
#
