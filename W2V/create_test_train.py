# Used to split data into train and test sets using keep-one-back validation
# Most recently rated film kept back

import pandas as pd
import pickle
from utils import get_user_interactions

ratings = pd.read_csv("./original_data/ratings.csv")
original_ids = pd.read_csv("./original_data/movies.csv")
titles = pd.read_csv("./cleaned_data/titles_index.csv")

ratings["rank_latest"] = ratings.groupby(["userId"])["timestamp"].rank(
    method="first", ascending=False
)

train_ratings = ratings[ratings["rank_latest"] != 1]
test_ratings = ratings[ratings["rank_latest"] == 1]

user_interacted_items = get_user_interactions(titles, train_ratings, original_ids)
user_test_items = get_user_interactions(titles, test_ratings, original_ids)

with open("cleaned_data/user_interacted_items.pkl", "wb") as f:
    pickle.dump(user_interacted_items, f)

with open("cleaned_data/user_test_items.pkl", "wb") as f:
    pickle.dump(user_test_items, f)
