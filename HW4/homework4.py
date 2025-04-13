# %% [markdown]
# # Homework 4: Fairness and bias interventions

# %% [markdown]
# ## Regression: Download the "wine quality" dataset:
# 
# https://archive.ics.uci.edu/dataset/186/wine+quality
# 
# ## Unzip the file "wine+quality.zip" to obtain:
# 
# - winequality.names
# - winequality-red.csv
# - winequality-white.csv

# %% [markdown]
# Predifine the answers:

# %%
answers = {}

# %% [markdown]
# ### Implement a  linear regressor using all continuous attributes (i.e., everything except color) to predict the wine quality. Use an 80/20 train/test split. Use sklearn’s `linear_model.LinearRegression`

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()

# Load datasets
winequality_red = pd.read_csv("winequality-red.csv", sep=';')
winequality_white = pd.read_csv("winequality-white.csv", sep=';')

# Concatenate the datasets
wine_data = pd.concat([winequality_red, winequality_white], axis=0).reset_index(drop=True)

# Set a random seed and split the train/test subsets
random_seed = 42
train_data, test_data = train_test_split(wine_data, test_size=0.2, random_state=random_seed)

# Display the train and test data
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Train the linear regression model
X_train = train_data.drop(columns=['quality'])
y_train = train_data['quality']
X_test = test_data.drop(columns=['quality'])
y_test = test_data['quality']

# normalize the dataset
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# %%
y_test

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%
# run linear regression here
model = LinearRegression()
model.fit(X_train_normalized, y_train)

# %% [markdown]
# 1. Report the feature with the largest coefficient value and the corresponding coefficient (not including any offset term).

# %%
feature = model.coef_.argmax()
corresponding_coefficient = model.coef_[feature]

# %%
answers['Q1'] = [feature, corresponding_coefficient]

# %% [markdown]
# 2. On the first example in the test set, determine which feature has the largest effect and report its effect (see "Explaining predictions using weight plots & effect plots").

# %%
largest_effect = X_test_normalized[0] * model.coef_
feature = (largest_effect).argmax()
corresponding_coefficient = largest_effect[feature]

feature, corresponding_coefficient

# %%
answers['Q2'] = [feature, corresponding_coefficient]

# %% [markdown]
# 3. (2 marks) Based on the MSE, compute ablations of the model including every feature (other than the offset). Find the most important feature (i.e., such that the ablated model has the highest MSE) and report the value of MSE_ablated - MSE_full.

# %%
MSE = mean_squared_error(y_test, model.predict(X_test_normalized))

# ablation
mse_diffs = {}

for i, feature_name in enumerate(X_train.columns):
    X_train_ablated = np.delete(X_train_normalized, i, axis=1)
    X_test_ablated = np.delete(X_test_normalized, i, axis=1)
    ablated_model = LinearRegression()
    ablated_model.fit(X_train_ablated, y_train)
    ablated_MSE = mean_squared_error(y_test, ablated_model.predict(X_test_ablated))
    mse_diffs[feature_name] = ablated_MSE - MSE

max_diff = max(mse_diffs, key=mse_diffs.get)
max_diff_value = mse_diffs[max_diff]
max_diff, max_diff_value

# %%
most_important_feature = max_diff_value
mse_diff = max_diff_value

# %%
answers['Q3'] = [most_important_feature, mse_diff]

# %% [markdown]
# 4. (2 marks) Implement a full backward selection pipeline and report the sequence of MSE values for each model as a list (of increasing MSEs).

# %%
mse_list = # increasing MSEs, same length as feature vector

# %%
answers['Q4'] = mse_list 

# %% [markdown]
# 5. (2 marks) Change your model to use an l1 regularizer. Increasing the regularization strength will cause variables to gradually be removed (coefficient reduced to zero) from the model. Which is the first and the last variable to be eliminated via this process?

# %%


# %%
answers['Q5'] = [first_feature, last_feature]

# %% [markdown]
# ### Implement a classifier to predict the wine color (red / white), again using an 80/20 train/test split, and including only continuous variables.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
winequality_red = pd.read_csv("winequality-red.csv", sep=';')
winequality_white = pd.read_csv("winequality-white.csv", sep=';')

# Add a column to distinguish red and white wines
winequality_red['type'] = 0  # Red wine (encoded as 0)
winequality_white['type'] = 1  # White wine (encoded as 1)

# Concatenate the datasets
wine_data = pd.concat([winequality_red, winequality_white], axis=0)

# Separate features (and drop "quality" to get continuous variables) and target
X = wine_data.drop(columns=['quality', 'type'])  # Drop the target column
y = wine_data['type']  # Target column (wine type)

# Perform train/test split
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Display shapes of the resulting splits
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# %% [markdown]
# 6. Report the odds ratio associated with the first sample in the test set.

# %%
odds_ratio = 

# %%
answers['Q6'] = odds_ratio

# %% [markdown]
# 7. Find the 50 nearest neighbors (in the training set) to the first datapoint in the test set, based on the l2 distance. Train a classifier using only those 50 points, and report the largest value of e^theta_j (see “odds ratio” slides).

# %%
value = 

# %%
answers['Q7'] = value

# %%
with open("answers.txt", "w") as file:
    file.write(answers)


