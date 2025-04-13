# %% [markdown]
# ## HW 3: Fairness and Bias Interventions

# %% [markdown]
# ## Download the dataset
# 
# 1. Go to the [Adult Dataset webpage](https://archive.ics.uci.edu/dataset/2/adult).
# 2. Download and unzip the file in the same directory as this notebook.

# %%
# TODO: download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score
import numpy as np

# %%
## DATA LOADING ##
header = ["age",
          "workclass",
          "fnlwgt",
          "education",
          "education-num",
          "marital-status",
          "occupation",
          "relationship",
          "race",
          "sex",
          "capital-gain",
          "capital-loss",
          "hours-per-week",
          "native-country"]

values = {"workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
          "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
          "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
          "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
          "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
          "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
          "sex": ["Female", "Male"],
          "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
         }

# %%
def feat(d):
    f = [1]
    for h in header:
        if h in values:
            onehot = [0]*len(values[h])
            try:
                onehot[values[h].index(d[h])] = 1 # not efficient! Should make an index
            except Exception as e:
                # Missing value
                pass
            f += onehot
        else: # continuous
            try:
                f.append(float(d[h]))
            except Exception as e:
                # Missing value
                f.append(0) # Replacing with zero probably not perfect!
    return f

# %%
dataset = []
labels = []
a = open("adult.data", 'r')
for l in a:
    if len(l) <= 1: break # Last line of the dataset is empty
    l = l.split(", ") # Could use a csv library but probably no need to here
    dataset.append(dict(zip(header, l)))
    labels.append(l[-1].strip()) # Last entry in each row is the label

X = [feat(d) for d in dataset]
y = [inc == '>50K' for inc in labels]

X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(X, y, dataset, test_size=0.2, random_state=42)

# %%
len(X), len(X_train), len(X_test), len(d_train), len(d_test)

print(X_train[0])
print(y_train[0])
print(d_train[0])

# %%
answers = {}

# %% [markdown]
# ## 3.1
# 
# #### (1 point)
# 
# Implement a logistic regression classification pipeline using an `80/20` test split. Use a regularization value of $C = 1$.
# 
# Treat “sex” as the “sensitive attribute” i.e., $z=1$ for females and $z=0$ for others.
# 
# **Report:** The discrimination in the dataset (see "pre-processing" module).

# %%
def pipeline(X_tr, y_tr):
    reg = LogisticRegression(C = 1, max_iter = 1000)
    return reg.fit(X_tr, y_tr)

# %%
y_train = np.array(y_train)

# %%
def discrimination_score(datapoints, labels):
    labels = np.array(labels)
    z_1_indices = [index for index, dict in enumerate(datapoints) if dict['sex'] == 'Male']
    z_0_indices = [index for index, dict in enumerate(datapoints) if dict['sex'] == 'Female']
    p_y_1_z_1 = np.mean(labels[z_1_indices])
    p_y_1_z_0 = np.mean(labels[z_0_indices])
    return p_y_1_z_1 - p_y_1_z_0

dataset_discrimination = discrimination_score(d_train, y_train)
print(f'Dataset discrimination: {dataset_discrimination:.6f}')

# %%
answers['Q1'] = dataset_discrimination

# %% [markdown]
# ## 3.2
# 
# #### (1 point)
# 
# **Report:** The discrimination of the classifier.

# %%
mod = pipeline(X_train, y_train)
preds_train_q2 = mod.predict(X_train)
classifier_discrimination_q2 = discrimination_score(d_train, preds_train_q2)

print(f'Classifier discrimination (Q2): {classifier_discrimination_q2:.6f}')

# %%
answers['Q2'] = classifier_discrimination_q2

# %% [markdown]
# ## 3.3
# #### (1 point)
# 
# Implement a "massaging" approach that improves the discrimination score by at least 3\%.
# 
# 
# **Report:** The new discrimination score.

# %%
probs = mod.predict_proba(X_train)[:, -1]
z_1_indices = [index for index, dict in enumerate(d_train) if dict['sex'] == 'Male']
z_0_indices = [index for index, dict in enumerate(d_train) if dict['sex'] == 'Female']
female_neg = [i for i in z_0_indices if y_train[i] == 0]
male_pos = [i for i in z_1_indices if y_train[i] == 1]
probs = mod.predict_proba(X_train)[:, 1]


# %%
# find promotion candidates


promote_candidates = sorted(female_neg, key=lambda x: probs[x], reverse=True)
demote_candidates = sorted(male_pos, key=lambda x: probs[x])
y_train_fixed_q3 = y_train
for i in promote_candidates[:100]:
    y_train_fixed_q3[i] = 1
for i in demote_candidates[:100]:
    y_train_fixed_q3[i] = 0

# train model
model_q3 = pipeline(X_train, y_train_fixed_q3)
preds_train_q3 = model_q3.predict(X_train)
classifier_discrimination_q3 = discrimination_score(d_train, preds_train_q3)

print(f'Classifier discrimination (Q3):\t {classifier_discrimination_q3:.6f}')
print(f'Classifier relative improvement (Q3):\t {100*(classifier_discrimination_q2 - classifier_discrimination_q3) / classifier_discrimination_q3:.6f}%')

# %%
answers['Q3'] = classifier_discrimination_q3

# %% [markdown]
# ## 3.4
# 
# #### (2 points)
# 
# Implement a "reweighting" approach that improves the discrimination score by at least 3%; report the new discrimination score.
# 
# **Report:** The new discrimination score.

# %%
weights = np.zeros(len(d_train))
len(weights)


# %%
# observed counts
p_z_1_y_1 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if dictionary['sex'] == 'Female' and y_train[index] == 1]) / len(d_train)
p_z_1_y_0 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if dictionary['sex'] == 'Female' and y_train[index] == 0]) / len(d_train)
p_z_0_y_1 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if dictionary['sex'] == 'Male' and y_train[index] == 1]) / len(d_train)
p_z_0_y_0 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if dictionary['sex'] == 'Male' and y_train[index] == 0]) / len(d_train)
p_z_0 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if y_train[index] == 0]) / len(d_train)
p_z_1 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if y_train[index] == 1]) / len(d_train)
p_y_0 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if y_train[index] == 0]) / len(d_train)
p_y_1 = len([(index, dictionary) for index, dictionary in enumerate(d_train) if y_train[index] == 1]) / len(d_train)

weights = np.zeros(len(d_train))
y_train_q4 = y_train

for i, d in enumerate(d_train):
    if y_train[i] == 1 and d['sex'] == 'Female':
        weights[i] = p_y_1 * p_z_1 / p_z_1_y_1
    elif y_train[i] == 0 and d['sex'] == 'Female':
        weights[i] = p_y_0 * p_z_1 / p_z_1_y_0
    elif y_train[i] == 1 and d['sex'] == 'Male':
        weights[i] = p_y_1 * p_z_0 / p_z_0_y_1
    else:
        weights[i] = p_y_0 * p_z_0 / p_z_0_y_0

len(weights)

# %%
model_q4 = LogisticRegression(C=1, max_iter=1000)
model_q4.fit(X_train, y_train, sample_weight=weights)
preds_train_q4 = model_q4.predict(X_train)
classifier_discrimination_q4 = discrimination_score(d_train, preds_train_q4)

print(f'Classifier discrimination (Q4):\t {classifier_discrimination_q4:.6f}')
print(f'Classifier relative improvement (Q4):\t {100*(classifier_discrimination_q2 - classifier_discrimination_q4) / classifier_discrimination_q2:.6f}%')

# %%
answers['Q4'] = classifier_discrimination_q4

# %% [markdown]
# ## 3.5
# 
# #### (2 points)
# 
# Implement a "post processing" (affirmative action) policy. Lowering per-group thresholds will increase both the (per-group) FPR and the (per-group) TPR. For whichever group has the lower TPR, lower the threshold until the TPR for both groups is (as close as possible to) equal. Report the rates (TPR_0, TPR_1, FPR_0, and FPR_1) for both groups.
# 
# **Report:** The TPR and FPR rates for both groups as a list: `[TPR_0, TPR_1, FPR_0, FPR_1]`.
# 

# %%
def get_rates(y_true, y_pred, group_indices):
    predicted_true = sum(1 for i in group_indices if y_true[i] == 1 and y_pred[i] == 1)
    predicted_false = sum(1 for i in group_indices if y_true[i] == 0 and y_pred[i] == 1)
    true_positive = sum(1 for i in group_indices if y_true[i] == 1)
    true_negative = sum(1 for i in group_indices if y_true[i] == 0)
    tpr = predicted_true / true_positive if true_positive > 0 else 0
    fpr = predicted_false / true_negative if true_negative > 0 else 0
    return tpr, fpr

mod = LogisticRegression(C=1, max_iter=1000)
mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)


# %%
female_indices_test = [i for i, d in enumerate(d_test) if d['sex'] == 'Female']
male_indices_test = [i for i, d in enumerate(d_test) if d['sex'] == 'Male']
len(female_indices_test), len(d_test), len(y_test)

# %%
tpr, fpr = get_rates(y_test, y_pred, female_indices)
tpr, fpr

# %%
male_thresh = 0.5
female_thresh_q5 = ...
ans_q5 = [None, None, None, None]  # [TPR_male, TPR_female, FPR_male, FPR_female]

print(ans_q5)

# %%
answers['Q5'] = ans_q5

# %% [markdown]
# ## 3.6
# 
# #### (1 point)
# 
# Modify the solution from Q5 to exclude the sensitive attribute ($z$) from the classifier’s feature vector. Implement the same strategy as in Q5.
# 
# **Report:** The TPR and FPR rates for both groups as a list: `[TPR_0, TPR_1, FPR_0, FPR_1]`.
# 

# %%
male_thresh = 0.5
female_thresh_q6 = ...
ans_q6 = [None, None, None, None]  # [TPR_male, TPR_female, FPR_male, FPR_female]

print(ans_q6)

# %%
answers['Q6'] = ans_q6

# %% [markdown]
# ## 3.7
# 
# #### (1 point)
# 
# Again modifying the solution from Q5, train two separate classifiers, one for $z=0$ and one for $z=1$. Implement the same strategy as in Q5.
# 
# **Report:** The TPR and FPR rates for both groups as a list: `[TPR_0, TPR_1, FPR_0, FPR_1]`.

# %%
# [YOUR CODE HERE]

model_male_q7 = ...
model_female_q7 = ...

scores_q7 = ...


male_thresh = 0.5
female_thresh_q7 = ...
ans_q7 = [None, None, None, None]  # [TPR_male, TPR_female, FPR_male, FPR_female]

# %%
answers['Q7'] = ans_q7

# %% [markdown]
# ## Saving Answers

# %%
import json

# %%
# extra step to make things serializable

with open('answers.txt', 'w' ) as f:
    json.dump(answers, f, indent=2)

# %%


# %%



