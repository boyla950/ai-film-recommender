# Evaluation function designed to calculate the Hit Rate @ 10 and Diversety of the system
# This can take some time to run due to the amount of data used to calculate metrics.
# Code modified from https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import pickle
import random

titles = pd.read_csv('./cleaned_data/titles_index.csv')
all_movieIds = titles.index.unique()


with open("user_vectors.pkl","rb") as f:
    user_vecs = pickle.load(f)

with open("movie_vectors.pkl","rb") as f:
    movie_vecs = pickle.load(f)

with open("cleaned_data/user_interacted_items.pkl","rb") as f:
    user_interacted_items = pickle.load(f)

with open("cleaned_data/user_test_items.pkl","rb") as f:
    user_test_items = pickle.load(f)


hits = []
for u in user_test_items:
    i = user_test_items[u]
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movieIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + i
    test_movie_vecs = [movie_vecs[index] for index in test_items]
    cosine_sim = cosine_similarity(user_vecs[u].reshape(1, -1), test_movie_vecs)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:10]
    sim_scores = [test_items[score[0]] for score in sim_scores]
    
    top10_items = sim_scores[0:10]
    
    if i[0] in top10_items:
        hits.append(1)
    else:
        hits.append(0)

hit_rate = np.average(hits)
        
print(f"The Hit Rate @ 10 is {hit_rate}")   
#expected value: ~0.296

sim10 = []

random_users = random.sample(list(user_vecs),1000)

for u in random_users:
    
        
    cosine_sim = cosine_similarity(user_vecs[u].reshape(1, -1), movie_vecs)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score[0] for score in sim_scores[:100] if score[0] not in user_interacted_items[u]]
    recs = sim_scores[0:10]


    sim = 0
    count = 0

    for i in range(len(recs)):

        for j in range(i + 1, len(recs) - 1):

            a = recs[i]
            b = recs[j]

            sim += cosine_similarity(movie_vecs[a].reshape(1, -1), movie_vecs[b].reshape(1, -1))[0][0]
            count += 1

    sim10.append(sim/count)

diversity = 1 - np.average(sim10)

print(f"The average Diversity across the top 10 recommendations is: {diversity}") 
#expected value: ~0.397