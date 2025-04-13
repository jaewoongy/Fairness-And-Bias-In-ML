############################################################################
# Modify code however you want, as long as you provide the functions below #
############################################################################

from sklearn import linear_model
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

random.seed(0)

#######################
# Part 1: Accuracy    #
#######################

# Baseline: Trivial model that includes a single feature, and the sensitive attribute

def p1model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight='balanced', 
        random_state=42
    )

    
# d: a dictionary describing a training instance (excluding the sensitive attribute)
# z: sensitive attribute (True if married)
# return: a feature vector (list of floats)
def p1feat(d,z):
    return [float(v) for v in d.values()] + [1.0*z]

#########################################
# Part 2: Dataset-based intervention    #
#########################################

# Baseline: Just double all of the instances with a positive sensitive attribute

def p2model():
    return p1model()

# data: the dataset, which is a list of tuples of the form (d,z,l)
# d: feature dictionary (excluding sensitive attribute)
# z: sensitive attributes
# l: label
# return: a dataset in the same form
def p2data(data):
    np.random.seed(42)
    X_train = [p3feat(d) for d, _, _ in data]
    y_train = [l for _, _, l in data]
    
    model = p1model()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_train)[:, 1]
    
    z1_pos_high = [i for i, (_, z, l) in enumerate(data) if z == 1 and l == 1 and probs[i] < 0.58]
    z0_neg_high = [i for i, (_, z, l) in enumerate(data) if z == 0 and l == 0 and probs[i] > 0.42] # .44 is best
    
    demote_candidates = z1_pos_high[:0]
    promote_candidates = random.sample(z0_neg_high[:500], 250)
    
    newd = []
    for i, (d, z, l) in enumerate(data):
        new_l = l
        if i in promote_candidates:
            new_l = 1
        elif i in demote_candidates:
            new_l = 0
        newd.append((d, z, new_l))
    
    return newd
    
#########################################
# Problem 3: Model-based intervention   #
#########################################

# Baseline: Give instances with z=0 twice the sample weight in a logistic regressor

def p3feat(d):
    return [float(v) for v in d.values()]

# data: the dataset, which is a list of tuples of the form (d,z,l)
# d: feature dictionary (excluding sensitive attribute)
# z: sensitive attributes
# l: label
# return: a model (already fit, so that receiver can call mod.predict)
def p3model(data):
    weights = []
    for _,z,_ in data:
        if z: weights.append(2)
        else: weights.append(1) # Assign higher instance weights to instances with z=0
    X_train = [p3feat(d) for d,_,_ in data]
    y_train = [l for _,_,l in data]
    mod = p1model()
    # You can use any model you want, though it must have a "predict" function which takes a feature vector
    mod.fit(X_train, y_train, sample_weight = weights)
    return mod

###########################################
# Problem 4: Post-processing intervention #
###########################################

# Baseline: Perturb per-group thresholds by a bit

# test_scores: scores (probability estimates) for your classifier from Part 1
# dTest: the test data (list of dictionaries, i.e., just the features)
# zTest: list of sensitive attributes (list of bool)
# return: list of predictions (list of bool)
def p4labels(test_scores, dTest, zTest):
    scores = np.array(test_scores)
    z_array = np.array(zTest)
    
    z0_scores = scores[~z_array]
    z1_scores = scores[z_array]
    
    thresholds = np.linspace(0.3, 0.7, 21)
    best_thresholds = [0.5, 0.5]
    best_diff = float('inf')
    
    for t0 in thresholds:
        for t1 in thresholds:
            ppr_z0 = np.mean(z0_scores >= t0) if len(z0_scores) > 0 else 0
            ppr_z1 = np.mean(z1_scores >= t1) if len(z1_scores) > 0 else 0
            
            diff = abs(ppr_z0 - ppr_z1)
            
            if diff < best_diff:
                best_diff = diff
                best_thresholds = [t0, t1]
    
    threshold0, threshold1 = best_thresholds
    
    predictions = []
    for s, z in zip(test_scores, zTest):
        if not z:
            predictions.append(1 if s > threshold0 else 0)
        else:
            predictions.append(1 if s > threshold1 else 0)
    
    return predictions


########################################################
# Problem 5: Optimize p-rule, subject to accuracy > X% #
########################################################

# Baseline: Reuse solution from Part 1 (i.e., no improvement in model fairness)

# dataTrain: the dataset, which is a tuple of (ds,z,l) (as in other functions)
# dTest: the test data (list of dictionaries, i.e., just the features)
# zTest: the test sensitive attributes (list of bool)
def p5(dataTrain, dTest, zTest):
    X_train = [p3feat(d) for d, _, _ in dataTrain]
    z_train = [z for _, z, _ in dataTrain]
    y_train = [l for _, _, l in dataTrain]
    
    weights = []
    for z, _ in zip(z_train, y_train):
        if z == 0:
            weights.append(1.5)
        else:
            weights.append(1.0)

    mod = p1model()
    mod.fit(X_train, y_train)
    X_test = [p3feat(d) for d in dTest]
    test_scores = mod.predict_proba(X_test)[:, 1]
    
    z_array = np.array(zTest)
    z0_scores = test_scores[~z_array]
    z1_scores = test_scores[z_array]
    
    thresholds = np.linspace(0.3, 0.7, 31)
    best_thresholds = [0.5, 0.5]
    best_diff = float('inf')
    
    for t0 in thresholds:
        for t1 in thresholds:
            ppr_z0 = np.mean(z0_scores >= t0) if len(z0_scores) > 0 else 0
            ppr_z1 = np.mean(z1_scores >= t1) if len(z1_scores) > 0 else 0
            
            diff = abs(ppr_z0 - ppr_z1)
            
            if diff < best_diff:
                best_diff = diff
                best_thresholds = [t0, t1]
    
    threshold0, threshold1 = best_thresholds
    
    predictions = []
    for s, z in zip(test_scores, zTest):
        if not z:
            predictions.append(1 if s >= threshold0 else 0)
        else:
            predictions.append(1 if s >= threshold1 else 0)
    
    return predictions

