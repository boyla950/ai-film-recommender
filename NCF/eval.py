# Evaluation function designed to calculate the Hit Rate @ 10 and Diversety of the system
# This can take some time to run due to the amount of data used to calculate metrics.
# Code modified from https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems

import numpy as np
import pandas as pd
import torch
import pickle
from models import NCF
from sklearn.metrics.pairwise import cosine_similarity
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ratings = pd.read_csv("./data/ratings.csv")
movies = pd.read_csv("./data/movies.csv")

ratings["rank_latest"] = ratings.groupby(["userId"])["timestamp"].rank(
    method="first", ascending=False
)

test_ratings = ratings[ratings["rank_latest"] == 1]

test_ratings = test_ratings[["userId", "movieId", "rating"]]

all_movieIds = ratings["movieId"].unique()


num_users = ratings["userId"].max() + 1
num_items = ratings["movieId"].max() + 1


if device == torch.device("cuda"):
    checkpoint = torch.load("./NCF_trained.pth")
else:
    checkpoint = torch.load("./NCF_trained.pth", map_location=torch.device("cpu"))

model = NCF(num_users, num_items).to(device)

model.load_state_dict(checkpoint["state_dict"])
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()

# User-item pairs for testing
test_user_item_set = set(zip(test_ratings["userId"], test_ratings["movieId"]))

# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby("userId")["movieId"].apply(list).to_dict()

hits = []
for (u, i) in test_user_item_set:
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movieIds) - set(interacted_items)

    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]

    predicted_labels = np.squeeze(
        model(torch.tensor([u] * 100).to(device), torch.tensor(test_items).to(device))
        .detach()
        .cpu()
        .numpy()
    )

    top10_items = [
        test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()
    ]

    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)

hit_rate = np.average(hits)
        
print(f"The Hit Rate @ 10 is {hit_rate}")   
#expected value: ~0.59


titles = pd.read_csv("../W2V/cleaned_data/titles_index.csv")

with open("../W2V/movie_vectors.pkl","rb") as f:
    movie_vecs = pickle.load(f)

test_user_item_set = random.sample(list(test_user_item_set),1000)

sim10 = []

for (u, _) in test_user_item_set:
    interacted_items = user_interacted_items[u]
    test_items = list(set(all_movieIds) - set(interacted_items))

    predicted_labels = np.squeeze(
        model(torch.tensor([u] * len(test_items)).to(device), torch.tensor(test_items).to(device))
        .detach()
        .cpu()
        .numpy()
    )

    top10_items = [
        test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()
    ]

    ncf_recs = movies[movies["movieId"].isin(top10_items)]["title"].head(10).tolist()

    vec_indices = titles[titles['title'].isin(ncf_recs)].index.tolist()

    sim = 0
    count = 0

    for i in range(len(vec_indices)):

        for j in range(i + 1, len(vec_indices) - 1):

            a = vec_indices[i]
            b = vec_indices[j]

            sim += cosine_similarity(movie_vecs[a].reshape(1, -1), movie_vecs[b].reshape(1, -1))[0][0]
            count += 1

    sim10.append(sim/count)

diversity = 1 - np.average(sim10)

print(f"The average Diversity across the top 10 recommendations is: {diversity}") 
#expected value: ~0.59