# Testing fuction to check 'accuracy' of movie vectors
# This is not an accurate metric of performance but simply 
# designed to eyeball that the model works as expected

import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

titles = pd.read_csv('./cleaned_data/titles_index.csv')

with open("movie_vectors.pkl","rb") as f:
    movie_vecs = pickle.load(f)

# movie vector 818 corresponds to Francis Ford Coppolla's 'The Godfather' (1972)

cosine_sim = cosine_similarity(movie_vecs[818].reshape(1, -1), movie_vecs)
scores = list(enumerate(cosine_sim[0]))
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
top_10 = sorted_scores[1:11]
top_10 = [score[0] for score in top_10]

print(titles['title'].iloc[top_10].head(10))

# returns :
# 1880     The Godfather: Part III (1990)
# 1158      The Godfather: Part II (1974)
# 3565            The Conversation (1974)
# 1566               The Rainmaker (1997)
# 1145              Apocalypse Now (1979)
# 3938            Gardens of Stone (1987)
# 15263            The Rain People (1969)
# 6180                 Dementia 13 (1963)
# 5905          One from the Heart (1982)
# 8741                 Rumble Fish (1983)

# Suggests the model is working as expected

