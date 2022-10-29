import pandas as pd
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open("user_vectors.pkl","rb") as f:
    user_vecs = pickle.load(f)

with open("movie_vectors.pkl","rb") as f:
    movie_vecs = pickle.load(f)

with open("cleaned_data/user_interacted_items.pkl","rb") as f:
    user_interacted_items = pickle.load(f)

titles = pd.read_csv('./cleaned_data/titles_index.csv')


def get_recommendations(user_id):

    
    user_vec = user_vecs[user_id]
    cosine_sim = cosine_similarity(user_vec.reshape(1, -1), movie_vecs)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    sim_scores = [score[0] for score in sim_scores]

    return titles['title'].iloc[sim_scores].head(10).tolist()


def update_user(user_id, movie_name):

    try:
        movie_id = titles[titles['title'] == movie_name].index[0]
    except:
        movie_id = None

    if movie_id is not None:

        movie_vec = movie_vecs[movie_id]

        user_vecs[user_id] = ((user_vecs[user_id] * len(user_interacted_items[user_id])) + movie_vec) / (len(user_interacted_items[user_id]) + 1)
        user_interacted_items[user_id].append(movie_id)
       
        with open('cleaned_data/user_interacted_items.pkl', 'wb') as f:
            pickle.dump(user_interacted_items, f)
        
        with open('user_vectors.pkl', 'wb') as f:
            pickle.dump(user_vecs, f)

        return 'Movie added successfully!'

    return 'Sorry, that movie doesn\'t appear on our database :( Have you tried using the full title? (e.g. \'Toy Story (1995)\')'

print()

print('Hello, welcome to the movie reccomendation system!!!')

print('This system works by using looking at the films you have already watched and')
print('learning what you like using a Word2Vec NLP model on the films metadata.')
print('It then recommends films that best fit with your user profile!')

print()

uid = input('Please enter your user ID: ')

print()

print(f"Welcome user {uid}! Would you like to: ")

done = False

while not done:

    print()

    print("A) Add a movie to your account?")
    print("B) Get some recommendations?")
    print("Choose any other input to exit")

    action = input('Please type either A or B: ')

    print()

    if action == 'A':

        movie_name = input('Please enter the name of the movie: ')

        response = update_user(int(uid), movie_name)

        print()

        print(response)

    elif action == 'B':

        recs = get_recommendations(int(uid))

        print()
        print('Based on your previous watches, we recommend:')

        for rec in recs:
            print(rec)

    else:

        print()

        print('exiting...')

        sys.exit()