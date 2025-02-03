# %% [markdown]
# ## Homework 2: Intro to bias and fairness

# %% [markdown]
# Download the German Credit dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
# (use the “numeric” version of the data)
# 
# Implement a (logistic regression) classification pipeline using an 80/20 test split. Use a regularization value of C = 1.
# 
# Treat the 20th feature (i.e., feat[19] in the numeric data, which is
# related to housing) as the “sensitive attribute” i.e., z=1 if the feature value is 1.

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression

german_credit = np.loadtxt('german.data-numeric')
attrs = german_credit[:, :-1] 
labels = 2 - german_credit[:, -1] # (1 = Good,  2 = Bad) -> (0=Bad, 1=good)

split_point = 800
X_train, X_test = attrs[:split_point], attrs[split_point:]
y_train, y_test = labels[:split_point], labels[split_point:]

sensitive_attribute = 19

model = LogisticRegression(C=1.001, max_iter=1000, random_state=42)

# %% [markdown]
# 1. Report the prevalence in the test set.

# %%
prevalence = np.mean(y_test)
prevalence

# %% [markdown]
# 2. Report the per-group prevalence for z=0 and z=1.

# %%
prevalence_0 = np.mean(y_test[X_test[:, 19] == 0])
prevalence_1 = np.mean(y_test[X_test[:, 19] == 1])
prevalence_0, prevalence_1

# %% [markdown]
# 3. What is the demographic parity (expressed as a ratio between z=0 and z=1) for your classifier on the test set?

# %%
y_pred = model.fit(X_train, y_train).predict(X_test)
y_pred_0 = y_pred[X_test[:, sensitive_attribute] == 0]
y_pred_1 = y_pred[X_test[:, sensitive_attribute] == 1]
y_test_0 = y_test[X_test[:, sensitive_attribute] == 0]
y_test_1 = y_test[X_test[:, sensitive_attribute] == 1]
parity = np.mean(y_pred_0) / np.mean(y_pred_1)
parity

# %% [markdown]
# 4. Report TPR_0, TPR_1, FPR_0, and FPR_1 (see "equal opportunity" slides).

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

def get_metrics(y_true, y_pred):
    # might be useful to fill this helper function
    # feel free to ignore this though
    conf_mat = confusion_matrix(y_true, y_pred)
    tp = conf_mat[1][1]
    fn = conf_mat[1][0]
    tn = conf_mat[0][0]
    fp = conf_mat[0][1]
    return tp, fn, tn, fp

tp0, fn0, tn0, fp0 = get_metrics(y_test_0, y_pred_0)
tp1, fn1, tn1, fp1 = get_metrics(y_test_1, y_pred_1)

TPR_0 = tp0 / (tp0 + fn0)
TPR_1 = tp1 / (tp1 + fn1)

FPR_0 = fp0 / (fp0 + tn0)
FPR_1 = fp1 / (fp1 + tn1)

TPR_0, TPR_1, FPR_0, FPR_1

# %% [markdown]
# 5. Compute PPV_0, PPV_1, NPV_0, and NPV_1 (see "are fairness goals compatible" slides).

# %%
PPV_0 = tp0 / (tp0 + fp0)
PPV_1 = tp1 / (tp1 + fp1)

NPV_0 = tn0 / (tn0 + fn0)
NPV_1 = tn1 / (tn1 + fn1)

PPV_0, PPV_1, NPV_0, NPV_1

# %% [markdown]
# 6. Implement a "fairness through unawareness" classifier, i.e., don't use Z in your feature vector. Find the classifier coefficient which undergoes the largest (absolute value) change compared to the classifier with the feature included, and report its new coefficient.
# 

# %%
X_train_no_z = np.delete(X_train, sensitive_attribute, axis=1)
X_test_no_z = np.delete(X_test, sensitive_attribute, axis=1)

# %%

new_model = LogisticRegression(C=1, max_iter=1000, random_state=42)
new_model.fit(X_train_no_z, y_train)
y_pred_no_z = new_model.predict(X_test_no_z)

curr_coeff = new_model.coef_[0]
old_coeff = np.delete(model.coef_[0], 19)

#np.abs(curr_coeff - old_coeff).argmax()
biggest_change_idx = np.abs(curr_coeff - old_coeff).argmax()

new_coeff = new_model.coef_[0][biggest_change_idx]

# %% [markdown]
# 7. Report the demographic parity of the classifier after implementing the above intervention.

# %%

new_y_pred_0 = y_pred_no_z[X_test[:, 19] == 0]
new_y_pred_1 = y_pred_no_z[X_test[:, 19] == 1]

new_parity = np.mean(new_y_pred_0) / np.mean(new_y_pred_1)
new_parity


# %% [markdown]
# 8. Report the Generalized False Positive Rate and Generalized False Negative Rate of your original (i.e., not the one with z excluded).
# 

# %%
y_prob = model.predict_proba(X_test)[:, 1]
len(y_prob), len(y_prob[y_pred == 0])

# %%
tp, fn, tn, fp = get_metrics(y_test, y_pred)
y_prob = model.predict_proba(X_test)[:, 1]

GFPR = 1 / (fp + tn) * np.sum(y_prob[y_test == 0])
GFNR = 1 / (fn + tp) * np.sum(1 - y_prob[y_test == 1])

GFPR, GFNR

# %% [markdown]
# 9. (harder, 2 marks) Changing the classifier threshold (much as you would to generate an ROC curve) will change the False Positive and False Negative rates for both groups (i.e., FP_0, FP_1, FN_0, FN_1). Implement a "fairness through unawareness" classifier like you did in Question 6 but instead use feature 19 (i.e., feat[18]) as the sensitive attribute. Using this classifier, find the (non-trivial) threshold that comes closest to achieving Treatment Equality, and report the corresponding values of FP_0, FP_1, FN_0, and FN_1.

# %%
threshold = 0.4

sensitive_attribute = 18
new_model = LogisticRegression(C=1, max_iter=1000, random_state=42)

sensitive_values = X_test[:, sensitive_attribute]
X_train_no_18 = np.delete(X_train, sensitive_attribute, axis=1)
X_test_no_18 = np.delete(X_test, sensitive_attribute, axis=1)
new_model.fit(X_train_no_18, y_train)
pred_prob = new_model.predict_proba(X_test_no_18)[:, 1]
y_pred_threshold = (pred_prob > threshold).astype(int)

y_pred_0 = y_pred_threshold[sensitive_values == 0]
y_pred_1 = y_pred_threshold[sensitive_values == 1]
y_test_0 = y_test[sensitive_values == 0]
y_test_1 = y_test[sensitive_values == 1]

tp0, fn0, tn0, fp0 = get_metrics(y_test_0, y_pred_0)
tp1, fn1, tn1, fp1 = get_metrics(y_test_1, y_pred_1)

# If this is 1.0 then the treatment is equal
print(fp1 * fn0 / fp0 * fn1)

# FP0 and FN0 are wrong
FP_0, FP_1, FN_0, FN_1 = fp0, fp1, fn0, fn1
FP_0, FP_1, FN_0, FN_1

# %%
answers = {
    "Q1": prevalence,           # prevalence
    "Q2": [prevalence_0, prevalence_1],  # prevalence_0, prevalence_1
    "Q3": parity,           # parity
    "Q4": [TPR_0, TPR_1, FPR_0, FPR_1], # TPR_0, TPR_1, FPR_0, FPR_1
    "Q5": [PPV_0, PPV_1, NPV_0, NPV_1], # PPV_0, PPV_1, NPV_0, NPV_1
    "Q6": [biggest_change_idx, new_coeff], # feature index, coefficient
    "Q7": new_parity,           # parity
    "Q8": [GFPR, GFNR],  # GFPR, GFNR
    "Q9": [FP_0, FP_1, FN_0, FN_1]  # FP_0, FP_1, FN_0, FN_1
}

# %%
# need to convert np to json
answers_json = {k: v.item() if isinstance(v, np.number) else 
                   [x.item() if isinstance(x, np.number) else x for x in v] 
                   if isinstance(v, list) else v 
                for k, v in answers.items()}


# %%
import json 
with open('answers_hw2.txt', 'w') as file:
    json.dump(answers_json, file)


# %%



