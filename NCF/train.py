# Training procedure for the NCF model
# MovieLensTrainDataset class used from https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems


import numpy as np
import pandas as pd
from models import NCF
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# class used from https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems
class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(
            zip(ratings["userId"], ratings["movieId"], ratings["rating"])
        )

        num_negatives = 4

        for u, i, _ in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

ratings = pd.read_csv("./data/ratings.csv")
new_ratings = pd.read_csv("./data/new_ratings.csv")
ratings = pd.concat([ratings, new_ratings], ignore_index=True)

ratings["rank_latest"] = ratings.groupby(["userId"])["timestamp"].rank(
    method="first", ascending=False
)

train_ratings = ratings[ratings["rank_latest"] != 1]
train_ratings = train_ratings[["userId", "movieId", "rating"]]
all_movieIds = ratings["movieId"].unique()

num_users = ratings["userId"].max() + 1
num_items = ratings["movieId"].max() + 1

all_movieIds = ratings["movieId"].unique()

model = NCF(num_users, num_items).to(device)

data = DataLoader(
    MovieLensTrainDataset(train_ratings, all_movieIds), batch_size=1024, num_workers=4
)

num_epochs = 200

NCF_opt = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):

    print(f"epoch: {epoch}")

    for batch_idx, batch in enumerate(data):

        user_input, item_input, labels = batch
        user_input, item_input, labels = (
            user_input.to(device),
            item_input.to(device),
            labels.to(device),
        )

        predicted_labels = model(user_input, item_input)

        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())

        loss.backward()
        NCF_opt.step()

        if batch_idx % len(data) == 500:

            print(f"batch: {batch_idx}")

checkpoint = {"state_dict": model.state_dict(), "optimizer": NCF_opt.state_dict()}

torch.save(model, "./NCF_trained.pth")

ratings.to_csv("./data/ratings.csv")

new_ratings = new_ratings[0:0]

new_ratings.to_csv("./data/new_ratings.csv")

