print("Loading...")

import datetime
import pandas as pd
import numpy as np
import torch
import sys
from models import NCF

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

movies = pd.read_csv("./data/movies.csv")
ratings = pd.read_csv("./data/ratings.csv")
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

user_interacted_items = ratings.groupby("userId")["movieId"].apply(list).to_dict()

def get_recommendations(u):

    interacted_items = user_interacted_items[u]
    not_interacted_items = list(set(all_movieIds) - set(interacted_items))

    predicted_labels = np.squeeze(
        model(
            torch.tensor([u] * len(not_interacted_items)),
            torch.tensor(not_interacted_items),
        )
        .detach()
        .numpy()
    )

    top10_items = [
        not_interacted_items[i]
        for i in np.argsort(predicted_labels)[::-1][0:10].tolist()
    ]

    ncf_recs = movies[movies["movieId"].isin(top10_items)]

    return ncf_recs["title"].head(10).tolist()


def update_user(user_id, movie_name):

    try:
        movie_id = movies[movies["title"] == movie_name]["movieId"].tolist()[0]
    except:
        movie_id = None

    if movie_id is not None:

        with open("data/new_ratings.csv", "a") as file_object:
            file_object.write(
                f"{user_id},{movie_id},0.0,{int(datetime.datetime.now().timestamp())}\n"
            )

        return "Movie added successfully! Your recommendations will be updated next time the model is re-trained."

    return "Sorry, that movie doesn't appear on our database :( Have you tried using the full title? (e.g. 'Toy Story (1995)')"


print()

print("Hello, welcome to the movie reccomendation system!!!")
print("This system uses a deep neural network to predict what films you might like")
print( "by comparing your viewing history with that of other users and recommending")
print("films watched by similar users to yourself!")

print()

uid = input("Please enter your user ID: ")

print()

print(f"Welcome user {uid}! Would you like to: ")

done = False

while not done:

    print()

    print("A) Add a movie to your account?")
    print("B) Get some recommendations?")
    print("Choose any other input to exit")

    action = input("Please type either A or B: ")

    print()

    if action == "A":

        movie_name = input("Please enter the name of the movie: ")

        response = update_user(int(uid), movie_name)

        print()

        print(response)

        pass

    elif action == "B":

        recs = get_recommendations(int(uid))

        print("Based on your previous watches, we recommend:")
        print()
        for rec in recs:
            print(rec)

    else:

        print()

        print("exiting...")

        sys.exit()
