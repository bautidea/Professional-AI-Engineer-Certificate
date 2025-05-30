# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
#
#
# # Decision Trees
#
#
# Estimated time needed: **25** minutes
#
#
# ## Objectives
#
# After completing this lab you will be able to:
#
# * Develop a classification model using Decision Tree Algorithm
# * Apply Decision Tree classification on a real world dataset.
#

# %% [markdown]
# ## Introduction
#
# This lab explores decision tree classification, a powerful machine learning technique for making data-driven decisions. You will learn to build, visualize, and evaluate decision trees using a real-world dataset. The dataset used in this lab is that of Drug prediction based on the health parameters of a patient.
#

# %% [markdown]
# ### Importing Libraries
# First, to make sure that the required libraries are available, execute the cell below.
#

# %%
import numpy as np
import warnings
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import pandas as pd
!pip install numpy == 2.2.0
!pip install pandas == 2.2.3
!pip install scikit-learn == 1.6.0
!pip install matplotlib == 3.9.3

# %% [markdown]
# Now import the required libraries for this lab.
#

# %%

%matplotlib inline

warnings.filterwarnings('ignore')

# %% [markdown]
# ### About the dataset
# Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug C, Drug X and Drug Y.
#
# Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are the Age, Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to.
#
# It is a sample of a multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of an unknown patient or to prescribe a drug to a new patient.
#

# %% [markdown]
# <div id="downloading_data">
#     <h2>Downloading the Data</h2>
#     To download the data, we will use !wget to download it from IBM Object Storage.
# </div>
#

# %%
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
my_data

# %% [markdown]
# ## Data Analysis and pre-processing
# You should apply some basic analytics steps to understand the data better. First, let us gather some basic information about the dataset.
#

# %%
my_data.info()

# %% [markdown]
# This tells us that 4 out of the 6 features of this dataset are categorical, which will have to be converted into numerical ones to be used for modeling. For this, we can make use of __LabelEncoder__ from the Scikit-Learn library.
#

# %%
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])
my_data

# %% [markdown]
# With this, you now have 5 parameters that can be used for modeling and 1 feature as the target variable.
# We can see from comparison of the data before Label encoding and after it, to note the following mapping.
# <br>
# For parameter 'Sex' : $M \rightarrow 1, F \rightarrow 0$ <br>
# For parameter 'BP' : $High \rightarrow 0, Low \rightarrow 1, Normal \rightarrow 2$<br>
# For parameter 'Cholesterol' : $High \rightarrow 0, Normal \rightarrow 1$
#

# %% [markdown]
# You can also check if there are any missing values in the dataset.
#

# %%
my_data.isnull().sum()

# %% [markdown]
# This tells us that there are no missing values in any of the fields.
#

# %% [markdown]
# To evaluate the correlation of the target variable with the input features, it will be convenient to map the different drugs to a numerical value. Execute the following cell to achieve the same.
#

# %%
custom_map = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'drugY': 4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
my_data

# %% [markdown]
# You can now use the __corr()__ function to find the correlation of the input variables with the target variable.
#
# #### Practice question
# Write the code to find the correlation of the input variables with the target variable and identify the features most significantly affecting the target.
#

# %%
corr_matrix = my_data.select_dtypes(include=[int, 'float64'])
corr_matrix.corr()

# %% [markdown]
# Most significant variables that affect the target are Na_to_K and probably BP.

# %% [markdown]
# # your code here
#

# %% [markdown]
# <details><summary>Click here for the solution</summary>
#
# ```python
# my_data.drop('Drug',axis=1).corr()['Drug_num']
# ```
#
# This shows that the drug recommendation is mostly correlated with the `Na_to_K` and `BP` features.
#
# </details>
#

# %% [markdown]
# We can also understand the distribution of the dataset by plotting the count of the records with each drug recommendation.
#

# %%
category_counts = my_data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()

# %% [markdown]
# This shows us the distribution of the different classes, clearly indicating that Drug X and Drug Y have many more records in comparison to the other 3.
#

# %% [markdown]
# ## Modeling
#

# %% [markdown]
# For modeling this dataset with a Decision tree classifier, we first split the dataset into training and testing subsets. For this, we separate the target variable from the input variables.
#

# %%
y = my_data['Drug']
X = my_data.drop(['Drug', 'Drug_num'], axis=1)

# %% [markdown]
# Now, use the __train_test_split()__ function to separate the training data from the testing data. We can make use of 30% of the data for testing and the rest for training the Decision tree.
#

# %%
X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.3, random_state=32)

# %% [markdown]
# You can now define the Decision tree classifier as __drugTree__ and train it with the training data.
#

# %%
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# %%
drugTree.fit(X_trainset, y_trainset)

# %% [markdown]
# ### Evaluation
#

# %% [markdown]
# Now that you have trained the decision tree, we can use it to generate the predictions on the test set.
#

# %%
tree_predictions = drugTree.predict(X_testset)

# %% [markdown]
# We can now check the accuracy of our model by using the accuracy metric.
#

# %%
print("Decision Trees's Accuracy: ",
      metrics.accuracy_score(y_testset, tree_predictions))

# %% [markdown]
# This means that the model was able to correctly identify the labels of 98.33%, i.e. 59 out of 60 test samples.
#

# %% [markdown]
# ### Visualize the tree
#
# To understand the classification criteria derived by the Decision Tree, we may generate the tree plot.
#

# %%
plot_tree(drugTree)
plt.show()

# %% [markdown]
# From this tree, we can derive the criteria developed by the model to identify the class of each training sample. We can interpret them by tracing the criteria defined by tracing down from the root to the tree's leaf nodes.
#
# For instance, the decision criterion for Drug Y is ${Na\_to\_K} \gt 14.627$.
#
# #### Practice Question:
# Along similar lines, identify the decision criteria for all other classes.
#

# %% [markdown]
# {
#     'drugA':0,
#     'drugB':1,
#     'drugC':2,
#     'drugX':3,
#     'drugY':4
# }
#
# DRUG A -> Na_to_K <= 14.6278 -> BP <= 0.5 -> Age <= 50.5
# DRUG B -> Na_to_K <= 14.6278 -> BP <= 0.5 -> Age > 50.5
# DRUG C -> Na_to_K <= 14.6278 -> BP > 0.5 -> Cholesterol <= 0.5 -> BP <= 1.5
# DRUG X -> Na_to_K <= 14.6278 -> BP > 0.5 -> Cholesterol <= 0.5 -> BP > 1.5
#        -> Na_to_K <= 14.6278 -> BP > 0.5 -> Cholesterol > 0.5

# %% [markdown]
# <details><summary>Click here for the solution</summary>
# Drug A : $Na\_to\_K <= 14.627, BP = High, Age <= 50.5$<br>
# Drug B : $Na\_to\_K <= 14.627, BP = High, Age > 50.5$ <br>
# Drug C : $Na\_to\_K <= 14.627, BP = Normal, Cholesterol = High$<br>
# Drug X : $Na\_to\_K <= 14.627, (BP = Low, Cholesterol = High) or (BP = Normal/Low, Cholesterol = Normal)$
# </details>
#

# %% [markdown]
# #### Practice Question:
#
# If the max depth of the tree is reduced to 3, how would the performance of the model be affected?
#

# %%
# your code here

# %% [markdown]
# <details><summary>Click here for the solution</summary>
#
# ```python
# drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
# drugTree.fit(X_trainset,y_trainset)
# tree_predictions = drugTree.predict(X_testset)
# print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))
# ```
#
# </details>
#

# %% [markdown]
# ### Congratulations! You're ready to move on to your next lesson!
#
# ## Author
# <a href="https://www.linkedin.com/in/abhishek-gagneja-23051987/" target="_blank">Abhishek Gagneja</a>
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
