# %% [markdown]
# ## Homework 5: Fairness and bias in application domains
# 

# %% [markdown]
# Use code from https://cseweb.ucsd.edu/~jmcauley/pml/ (Chapter 4) to build a simple recommender based on beer review data. You can use the “mostSimilarFast” function as your recommender, though will have to modify the data loader a little bit to use the beer dataset. You may use the 50,000 review dataset available here:
# https://datarepo.eng.ucsd.edu/mcauley_group/pml_data/beer_50000.json
# 
# You may also use code from the above link for simple utilities (e.g. Gini coefficient)

# %%
import ast
from collections import defaultdict
import math
import numpy as np


# %%
answers = {
    "Q1": None, 
    "Q2": None,
    "Q3": None,
    "Q4": None,
}

# %% [markdown]
# ### Loading the beer dataset

# %%
dataset = []
with open('beer_50000.json', 'r') as file:
    for line in file:
        dataset.append(ast.literal_eval(line))

# %%
dataset[0]

# %% [markdown]
# ### Q1

# %%
# Extract a few utility data structures
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataset:
    user,item = d['user/profileName'], d['beer/beerId']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user,item)] = d['review/overall']
    itemNames[item] = d['beer/name']

# %%
# your code here
def Jaccard(s1, s2):
   numer = len(s1.intersection(s2))
   denom = len(s1.union(s2))
   if denom == 0:
       return 0
   return numer / denom

def gini(z, samples=1000000):
    m = sum(z) / len(z)
    denom = 2 * samples * m
    numer = 0
    for _ in range(samples):
        i = np.random.choice(z)
        j = np.random.choice(z)
        numer += math.fabs(i - j)
    return numer / denom

def giniExact(z):
    m = sum(z) / len(z)
    denom = 2 * len(z)**2 * m
    numer = 0
    for i in range(len(z)):
        for j in range(len(z)):
            numer += math.fabs(z[i] - z[j])
    return numer / denom


# %%
def mostSimilarFast(i, N):
   similarities = []
   users = usersPerItem[i]
   candidateItems = set()
   for u in users:
       candidateItems = candidateItems.union(itemsPerUser[u])
   for i2 in candidateItems:
       if i2 == i:
           continue
       sim = Jaccard(users, usersPerItem[i2])
       similarities.append((sim,i2))
   similarities.sort(reverse=True)
   return similarities[:N]


# %%
first_100_items = sorted(list(usersPerItem.keys()))[:100]

all_recommendations = []
for item in first_100_items:
    similar_items = mostSimilarFast(item, 100)
    for _, rec_item in similar_items:
        all_recommendations.append(rec_item)

data_pop = {}
for item in usersPerItem:
    data_pop[item] = len(usersPerItem[item])
data_pop_values = list(data_pop.values())

rec_pop = {}
for item in all_recommendations:
    if item in rec_pop:
        rec_pop[item] += 1
    else:
        rec_pop[item] = 1
rec_pop_values = list(rec_pop.values())

q1_gini_data = giniExact(data_pop_values)
q1_gini_rec = giniExact(rec_pop_values)

answers['Q1'] = [q1_gini_data, q1_gini_rec]

# %%
answers['Q1'] = [q1_gini_data, q1_gini_rec]

# %%
assert(len(answers['Q1']) == 2)

# %% [markdown]
# ### Q2

# %%
import math

styles = {}
for d in dataset:
    style = d.get('beer/style', 'Unknown')
    styles[style] = styles.get(style, 0) + 1

sorted_styles = sorted(styles.items(), key=lambda x: x[1], reverse=True)
top10 = sorted_styles[:10]
top_styles = [s for s, _ in top10]

dataset_size = len(dataset)
P = {style: count / dataset_size for style, count in top10}
style_lookup = {d['beer/beerId']: d.get('beer/style', 'Unknown') for d in dataset}
recs = []
for item in list(usersPerItem.keys())[:100]:
    similar_items = mostSimilarFast(item, 100)
    recs.extend(rec for _, rec in similar_items)

rec_counts = {}
for item in recs:
    if item in style_lookup:
        style = style_lookup[item]
        rec_counts[style] = rec_counts.get(style, 0) + 1

alpha = 0.01
total_recs = len(recs) + 10 * alpha
Q = {s: (rec_counts.get(s, 0) + alpha) / total_recs for s in top_styles}

calibration = sum(P[s] * math.log2(P[s] / Q[s]) for s in top_styles if Q[s] > 0)

answers['Q2'] = calibration


# %%
answers['Q2']

# %%
answers['Q2'] = calibration

# %% [markdown]
# ### Q3

# %%
beer_abv = {}
for d in dataset:
    beer_id = d['beer/beerId']
    try:
        abv = d['beer/ABV']
        beer_abv[beer_id] = abv
    except:
        beer_abv[beer_id] = 0  

# %%
# Fill in the details of this function
# The skeleton is implemented as below to induce similar results
def mostSimilarMMR(i, N, lamb=0.5):
    users = usersPerItem[i]
    candidateItems = set()
    for u in users:
        candidateItems = candidateItems.union(itemsPerUser[u])
    
    abv_i = beer_abv.get(i, 0)
    
    all_similarities = {}
    for i2 in sorted(candidateItems):
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        all_similarities[i2] = sim
    
    selectedItems = set() 
    remainingItems = sorted(all_similarities.keys())
    
    while len(selectedItems) < N and remainingItems:
        # Select one item s.t. score is maximized 
        # score = lamb * first_part - (1-lamb) * second_part
        
        best_score = float('-inf')
        best_item = None
        
        for item in remainingItems: 
            first_part = all_similarities[item]
            
            second_part = None
            
            for _, j in sorted(selectedItems):
                abv_sim = -(beer_abv.get(item, 0) - beer_abv.get(j, 0))**2
                if not second_part or abv_sim > second_part:
                    second_part = abv_sim
            
            if not second_part:
                second_part = 0
            
            mmr_score = lamb * first_part - (1-lamb) * second_part
            
            if mmr_score > best_score: 
                best_score = mmr_score
                best_item = item
        
        if best_item is not None:
            selectedItems.add((best_score, best_item))
            remainingItems.remove(best_item)
    
    return sorted(list(selectedItems), key=lambda x: x[0], reverse=True)

# %%
q3_most_similar = mostSimilarMMR(dataset[0]['beer/beerId'], N=10, lamb=0.5)

# %%
q3_items = [i for s, i in q3_most_similar]
q3_scores = [s for s, i in q3_most_similar]

# %%
list(zip(q3_items, q3_scores))

# %%
answers['Q3'] = [q3_items, q3_scores]

# %%
assert(len(answers['Q3'][0]) == 10)

# %% [markdown]
# ### Q4

# %%
q4_answer = []

for lamb in [1, 0.8, 0.6, 0.4]:
    # your code
    all_relevance = []
    for d in dataset[:100]:
        # your code
        item_id = d['beer/beerId']
        recommendations = mostSimilarMMR(item_id, N=10, lamb=lamb)
        
        for _, rec_item in recommendations:
            ratings = []
            for user in usersPerItem[rec_item]:
                if (user, rec_item) in ratingDict:
                    ratings.append(ratingDict[(user, rec_item)])
            
            if ratings:
                all_relevance.append(sum(ratings) / len(ratings))
    
    if all_relevance:
        avg_relevance = sum(all_relevance) / len(all_relevance)
        q4_answer.append(avg_relevance)
    else:
        q4_answer.append(0)

answers['Q4'] = q4_answer

# %%
answers['Q4'] = q4_answer

# %%
assert(len(answers['Q4']) == 4)

# %% [markdown]
# # Results

# %%
answers

# %%
import json

with open("answers_hw5.txt", "w") as file:
    json.dump(answers, file, indent=4, default=str)

# %%
answers

# %%


# %%



