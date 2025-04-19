# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="300" alt="Skills Network Logo">
#     </a>
# </p>
#
#
# # Final Project: Building a Rainfall Prediction Classifier
# Estimated time needed: **60** minutes
#
#
# ## Objectives
#
# After completing this lab you will be able to:
#
# * Explore and perform feature engineering on a real-world data set
# * Build a classifier pipeline and optimize it using grid search cross validation
# * Evaluate your model by interpreting various performance metrics and visualizations
# * Implement a different classifier by updating your pipeline
# * Use an appropriate set of parameters to search over in each case
#
# ## Instruction(s)
#
# After completing the Notebook:
#
# * Download the notebook using **File** > **Download**.
# * This notebook will be then graded using **AI grader** in the subsequent section.
# * Copy/Paste your markdown responses in the subsequent **AI Mark assignment**.
#
#
#

# %% [markdown]
# # About The Dataset
# The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from [http://www.bom.gov.au/climate/dwo/](http://www.bom.gov.au/climate/dwo/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01).
#
# The dataset you'll use in this project was downloaded from Kaggle at [https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download&select=weatherAUS.csv)
# Column definitions were gathered from [http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)
#
# The dataset contains observations of weather metrics for each day from 2008 to 2017, and includes the following fields:
#
# | Field         | Description                                           | Unit            | Type   |
# | :------------ | :---------------------------------------------------- | :-------------- | :----- |
# | Date          | Date of the Observation in YYYY-MM-DD                 | Date            | object |
# | Location      | Location of the Observation                           | Location        | object |
# | MinTemp       | Minimum temperature                                   | Celsius         | float  |
# | MaxTemp       | Maximum temperature                                   | Celsius         | float  |
# | Rainfall      | Amount of rainfall                                    | Millimeters     | float  |
# | Evaporation   | Amount of evaporation                                 | Millimeters     | float  |
# | Sunshine      | Amount of bright sunshine                             | hours           | float  |
# | WindGustDir   | Direction of the strongest gust                       | Compass Points  | object |
# | WindGustSpeed | Speed of the strongest gust                           | Kilometers/Hour | object |
# | WindDir9am    | Wind direction averaged over 10 minutes prior to 9am  | Compass Points  | object |
# | WindDir3pm    | Wind direction averaged over 10 minutes prior to 3pm  | Compass Points  | object |
# | WindSpeed9am  | Wind speed averaged over 10 minutes prior to 9am      | Kilometers/Hour | float  |
# | WindSpeed3pm  | Wind speed averaged over 10 minutes prior to 3pm      | Kilometers/Hour | float  |
# | Humidity9am   | Humidity at 9am                                       | Percent         | float  |
# | Humidity3pm   | Humidity at 3pm                                       | Percent         | float  |
# | Pressure9am   | Atmospheric pressure reduced to mean sea level at 9am | Hectopascal     | float  |
# | Pressure3pm   | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascal     | float  |
# | Cloud9am      | Fraction of the sky obscured by cloud at 9am          | Eights          | float  |
# | Cloud3pm      | Fraction of the sky obscured by cloud at 3pm          | Eights          | float  |
# | Temp9am       | Temperature at 9am                                    | Celsius         | float  |
# | Temp3pm       | Temperature at 3pm                                    | Celsius         | float  |
# | RainToday     | If there was at least 1mm of rain today               | Yes/No          | object |
# | RainTomorrow  | If there is at least 1mm of rain tomorrow             | Yes/No          | object |
#
#

# %% [markdown]
# ## Install and import the required libraries
#

# %% [markdown]
# Exectue the following cells to install and import the necessary libraries.
#

# %%
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import pandas as pd
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn

# %%

# %% [markdown]
# ## Load the data
#

# %% [markdown]
# Execute the following cells to load the dataset as a pandas dataframe.
#

# %%
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

# %%
df.count()

# %% [markdown]
# Sunshine and cloud cover seem like important features, but they have a lot of missing values, far too many to impute their missing values.
#

# %% [markdown]
# ### Drop all rows with missing values
# To try to keep things simple we'll drop rows with missing values and see what's left
#

# %%
df = df.dropna()
df.info()

# %% [markdown]
# Since we still have 56k observations left after dropping missing values, we may not need to impute any missing values.
# Let's see how we do.
#

# %%
df.columns

# %% [markdown]
# ## Data leakage considerations
# Consider the descriptions above for the columns in the data set. Are there any practical limitations to being able to predict whether it will rain tomorrow given the available data?
#
# ## Points to note - 1
# List some of the features that would be inefficient in predicting tomorrow's rainfall. There will be a question in the quiz that follows based on this observation.
#

# %% [markdown]
# <details><summary>Click here for Hint</summary>
#
# Consider features that rely on the entire duration of today for their evaluation.
#
# </details>
#

# %% [markdown]
# If we adjust our approach and aim to predict today’s rainfall using historical weather data up to and including yesterday, then we can legitimately utilize all of the available features. This shift would be particularly useful for practical applications, such as deciding whether you will bike to work today.
#
# With this new target, we should update the names of the rain columns accordingly to avoid confusion.
#

# %%
inefficient_predictors = ['Location', 'MinTemp', 'MaxTemp']

# %%
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# %% [markdown]
# ## Data Granularity
#

# %% [markdown]
# Would the weather patterns have the same predictability in vastly different locations in Australia? I would think not.
# The chance of rain in one location can be much higher than in another.
# Using all of the locations requires a more complex model as it needs to adapt to local weather patterns.
# Let's see how many observations we have for each location, and see if we can reduce our attention to a smaller region.
#

# %% [markdown]
# ## Location selection
# You could do some research to group cities in the `Location` column by distance, which I've done for you behind the scenes.
# I found that Watsonia is only 15 km from Melbourne, and the Melbourne Airport is only 18 km from Melbourne.
# Let's group these three locations together and use only their weather data to build our localized prediction model.
# Because there might still be some slight variations in the weather patterns we'll keep `Location` as a categorical variable.
#

# %%
df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia',])]
df. info()

# %% [markdown]
# We still have 7557 records, which should be enough to build a reasonably good model.
# You could always gather more data if needed by partioning the data into similar locations or simplyby updating it from the source to include a larger time frame.
#
# ## Extracting a seasonality feature
# Now consider the `Date` column. We expect the weather patterns to be seasonal, having different predictablitiy levels in winter and summer for example.
# There may be some variation with `Year` as well, but we'll leave that out for now.
# Let's engineer a `Season` feature from `Date` and drop `Date` afterward, since it is most likely less informative than season.
# An easy way to do this is to define a function that assigns seasons to given months, then use that function to transform the `Date` column.
#

# %% [markdown]
# ### Create a function to map dates to seasons
#

# %%


def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# %% [markdown]
# ## Exercise 1: Map the dates to seasons and drop the Date column
# Complete the code:
# ```python
# # Convert the 'Date' column to datetime format
# df['Date'] = pd.to_datetime(...)
#
# # Apply the function to the 'Date' column
# df['Season'] = df['Date'].apply(date_to_season)
#
# df=df.drop(columns=...)
# df
# ```
#


# %%
# Write your response.
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df.drop(columns='Date', inplace=True)

# %% [markdown]
# Looks like we have a good set of features to work with.
#
# Let's go ahead and build our model.
#
# But wait, let's take a look at how well balanced our target is.
#

# %% [markdown]
# ## Exercise 2. Define the feature and target dataframes
# Complete the followng code:
# ```python
# X = df.drop(columns='...', axis=1)
# y = df['...']
# ```
#

# %%
# Write your response.
X = df.drop(columns='RainToday')
y = df['RainToday']

# %% [markdown]
# ## Exercise 3. How balanced are the classes?
# Display the counts of each class.
#
# Complete the following code:
# ```python
# ... .value_counts()
# ```
#

# %%
# Write your response.
y.value_counts(normalize=True)

# %% [markdown]
# ## Exercise 4. What can you conclude from these counts?
# - How often does it rain annualy in the Melbourne area?
# - How accurate would you be if you just assumed it won't rain every day?
# - Is this a balanced dataset?
# - Next steps?
#

# %% [markdown]
# 1- From 365 days, 24% are rainy days (88 days), so if we sparce that amount over 365 days -> 365 / 88 -> 4 times a year.
#
# 2- From 365 days, 76% are non-rainy days (277 days), so if i assume that 100% days will not rain, i would be correct 76% of timess.
#
# 3- The Dataset is highly unbalanced.
#
# 4- Split data, train, test, validate model.

# %% [markdown]
# ## Exercise 5. Split data into training and test sets, ensuring target stratification
#
# Complete the followng code:
# ```python
# X_train, X_test, y_train, y_test = train_test_split(..., ..., test_size=0.2, stratify=..., random_state=42)
# ```
#

# %%
# Write your response.
rnd_st = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=rnd_st, stratify=y, test_size=0.2)

# %% [markdown]
# ## Define preprocessing transformers for numerical and categorical features
# ## Exercise 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features
#
# Complete the followng code:
# ```python
# numeric_features = X_train.select_dtypes(include=['...']).columns.tolist()
# categorical_features = X_train.select_dtypes(include=['...', 'category']).columns.tolist()
# ```
#

# %%
# Write your response.
numeric_features = X_train.select_dtypes(
    include=[float, int]).columns.to_list()
categorical_features = X_train.select_dtypes(
    include=[object]).columns.to_list()

# %% [markdown]
# ### Define separate transformers for both feature types and combine them into a single preprocessing transformer
#

# %%
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals
categorical_transformer = Pipeline(
    steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# %% [markdown]
# ## Exercise 7. Combine the transformers into a single preprocessing column transformer
# Complete the followng code:
# ```python
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, ...),
#         ('cat', categorical_transformer, ...)
#     ]
# )
# ```
#

# %%
# Write your response.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %% [markdown]
# ## Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier
# Complete the following code:
# ```python
# pipeline = Pipeline(steps=[
#     ('preprocessor', ...),
#     ('...', RandomForestClassifier(random_state=42))
# ])
# ```
#

# %%
# Write your response.
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=rnd_st))
    ]
)

# %% [markdown]
# ### Define a parameter grid to use in a cross validation grid search model optimizer
#

# %%
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# %% [markdown]
# ### Pipeline usage in crossvalidation
# Recall that the pipeline is repeatedly used within the crossvalidation by fitting on each internal training fold and predicting on its corresponding validation fold
#

# %% [markdown]
# ## Perform grid search cross-validation and fit the best model to the training data
# ### Select a cross-validation method, ensuring target stratification during validation
#

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True)

# %% [markdown]
# ## Exercise 9. Instantiate and fit GridSearchCV to the pipeline
# Complete the followng code:
# ```python
# grid_search = GridSearchCV(..., param_grid, cv=..., scoring='accuracy', verbose=2)
# grid_search.fit(..., ...)
# ```
#

# %%
# Write your response.
grid_search = GridSearchCV(
    estimator=pipeline,
    cv=cv,
    param_grid=param_grid,
    verbose=2,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# %% [markdown]
# ### Print the best parameters and best crossvalidation score
#

# %%
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# %% [markdown]
# ## Exercise 10. Display your model's estimated score
# Complete the followng code:
# ```python
# test_score = grid_search.score(..., ...)
# print("Test set score: {:.2f}".format(test_score))
# ```
#

# %%
# Write your response.
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# %% [markdown]
# So we have a reasonably accurate classifer, which is expected to correctly predict about 84% of the time whether it will rain today in the Melbourne area.
# But careful here. Let's take a deeper look at the results.
#
# The best model is stored within the gridsearch object.
#

# %% [markdown]
# ## Exercise 11. Get the model predictions from the grid search estimator on the unseen data
# Complete the followng code:
# ```python
# y_pred = grid_search.predict(...)
# ```
#

# %%
# Write your response.
y_pred = grid_search.predict(X_test)

# %% [markdown]
# ## Exercise 12. Print the classification report
# Complete the followng code:
# ```python
# print("\nClassification Report:")
# print(...(y_test, y_pred))
# ```
#

# %%
# Write your response.
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## Exercise 13. Plot the confusion matrix
# Complete the followng code:
# ```python
# conf_matrix = ...(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=...)
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()
# ```
#

# %%
# Write your response.
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# Let's consider wether the results indicate a good predictor of rainfall.
# ## Points to note - 2
# What is the true positive rate? There will be a question on this in the assignment that follows.
#

# %%
TP = 184
FN = 174
TPR = TP / (TP + FN)
print(f'TPR = {TPR}')

# %% [markdown]
# <details><summary>Click here for Hints</summary>
#
# Consider the confusion matrix or the classification report and claculate the true positve rate given the information.
#
# </details>
#

# %% [markdown]
# ## Feature importances
# Recall that to obtain the categorical feature importances, we have to work our way backward through the modelling pipeline to associate the feature importances with their original input variables, not the one-hot encoded ones. We don't need to do this for the numeric variables because we didn't modify their names in any way.
# Remember we went from categorical features to one-hot encoded features, using the 'cat' column transformer.
#
# Let's get all of the feature importances and associate them with their transformed features
#

# %% [markdown]
# ## Exercise 14. Extract the feature importances
# Complete the followng code:
# ```python
# feature_importances = grid_search.best_estimator_['classifier']. ...
# ```
#

# %%
# Write your response.
feature_importances = grid_search.best_estimator_[
    'classifier'].feature_importances_

# %% [markdown]
# Now let's extract the feature importances and plot them as a bar graph.
#

# %%
# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_[
    'classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                              }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(
    f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

# %% [markdown]
# ## Point to note - 3
# Identify the most important feature for predicting whether it will rain based on the feature importance bar graph. There will be a question on this in the assignment that follows.
#

# %% [markdown]
# ## Try another model
# #### Some thoughts.
# In practice you would want to try out different models and even revisit the data analysis to improve
# your model's performance. Maybe you can engineer better features, drop irrelevant or redundant ones, project your data onto a dimensional feature space, or impute missing values to be able to use more data. You can also try a larger set of parameters to define you search grid, or even engineer new features using cluster analysis. You can even include the clustering algorithm's hyperparameters in your search grid!
#
# With Scikit-learn's powerful pipeline and GridSearchCV classes, this is easy to do in a few steps.
#
# ## Exercise 15. Update the pipeline and the parameter grid
# Let's update the pipeline and the parameter grid and train a Logistic Regression model and compare the performance of the two models. You'll need to replace the clasifier with LogisticRegression. We have supplied the parameter grid for you.
#
# Complete the following code:
# ```python
# # Replace RandomForestClassifier with LogisticRegression
# pipeline.set_params(...=LogisticRegression(random_state=42))
#
# # update the model's estimator to use the new pipeline
# grid_search.estimator = ...
#
# # Define a new grid with Logistic Regression parameters
# param_grid = {
#     # 'classifier__n_estimators': [50, 100],
#     # 'classifier__max_depth': [None, 10, 20],
#     # 'classifier__min_samples_split': [2, 5],
#     'classifier__solver' : ['liblinear'],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__class_weight' : [None, 'balanced']
# }
#
# grid_search.param_grid = ...
#
# # Fit the updated pipeline with LogisticRegression
# model.fit(..., ...)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# ```
#

# %%
# Write your response
pipeline.set_params(classifier=LogisticRegression(random_state=rnd_st))
grid_search.estimator = pipeline

param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search.param_grid = param_grid

grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

# %% [markdown]
# ###  Compare the results to your previous model.
# Display the clasification report and the confusion matrix for the new model and compare your results with the previous model.
#

# %%
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# What can you conclude about the model performances?
#

# %% [markdown]
# ## Points to note - 4
# Compare the accuracy and true positive rate of rainfall predictions between the LogisticRegression model and the RandomForestClassifier model.
#
# **Note: Make sure to provide the answer in the form of a list using either bullets or numbers.**
#
# There will be a question on this in the assignment that follows.
#

# %%
TP = 182
FN = 176
TPR = TP / (TP + FN)
print(f'TPR = {TPR}')

# %% [markdown]
# RandomForestClassifier:
# 1. Accuracy = 0.85
# 2. TPR = 0.51
# 3. TP = 184
# 4. TN = 1098
#
# LogisticRegression:
# 1. Accuracy = 0.83
# 2. TPR = 0.508
# 3. TP = 182
# 4. TN = 176

# %% [markdown]
# <details><summary>Click here for Hints</summary>
#
#    Compare the accuracy percentages of both the classifiers.
#
#    Provide the details of the number of correct predictions.
#
#    Provide the true positive rate of LogisticRegression Classifier.
#
#
# </details>
#

# %% [markdown]
#
# ### Congratulations! You've made it the end of your final project!
# Well done! You now have some great tools to use for tackling complex real-world problems with machine learning.
#
# ## Author
#
# <a href="https://www.linkedin.com/in/jpgrossman/" target="_blank">Jeff Grossman</a>
#
# ### Other Contributor(s)
#
# <a href="https://www.linkedin.com/in/abhishek-gagneja-23051987/" taget="_blank">Abhishek Gagneja</a>
#
# <!-- ## Changelog
#
# | Date | Version | Changed by | Change Description |
# |:------------|:------|:------------------|:---------------------------------------|
# | 2024-11-26 | 0.1  | Jeff Grossman    | Create lab |
#
#  -->
# <h3 align="center"> © IBM Corporation. All rights reserved. <h3/>
#

# %% [markdown]
#
#
