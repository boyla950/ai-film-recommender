import pandas as pd
import numpy as np
from ast import literal_eval
import nltk
from gensim.models import Word2Vec
import ast

import warnings

warnings.simplefilter("ignore")


def get_cast(credit_list):
    credit_list = ast.literal_eval(credit_list)
    cast_list = []
    for member in credit_list:
        if isinstance(member["name"], str):
            cast_list.append(member["name"].replace(" ", ""))
    cast_string = ""
    cast_list = cast_list[:3]
    for name in cast_list:
        cast_string = cast_string + name + " "
    return cast_string


def get_prod_company(pc):

    companies = ast.literal_eval(pc)
    if isinstance(companies, list) == 0:
        companies = []
    pc_list = []
    for company in companies:
        if isinstance(company["name"], str):
            pc_list.append(company["name"].replace(" ", ""))
    pc_string = ""
    pc_list = pc_list[:3]
    for company in pc_list:
        pc_string = pc_string + company + " "
    return pc_string


def get_director(credit_list):
    credit_list = ast.literal_eval(credit_list)
    for member in credit_list:
        if member["job"] == "Director":
            return member["name"].replace(" ", "")


def get_genres(genres):
    genres = ast.literal_eval(genres)
    genre_list = []
    if isinstance(genres, list):
        for genre in genres:
            genre_list.append(genre["name"].replace(" ", ""))
    else:
        genre_list.append(genres["name"])
    genre_list = genre_list[:3]
    genre_string = ""
    for name in genre_list:
        genre_string = genre_string + name + " "
    return genre_string


def get_desc(x):

    x = x.fillna("")
    x["title"] = x["title"] + " (" + x["year"] + ")"
    x["cast"] = x["cast"].apply(get_cast)
    x["director"] = x["crew"].apply(get_director)
    x["genres"] = x["genres"].apply(get_genres)
    x["production_company"] = x["production_companies"].apply(get_prod_company)
    x["desc"] = (
        x["cast"]
        + (2 * (" " + x["director"]))
        + " "
        + x["genres"]
        + " "
        + (2 * (x["production_company"]))
    )

    x["desc"] = x["desc"].fillna("")
    x.drop_duplicates(inplace=True, subset="title", keep="first")

    return x


def build_model(x):

    nltk.download("punkt")
    final_sentences = []

    for sentence in x["desc"]:
        new_sentence = nltk.tokenize.sent_tokenize(sentence)
        for s in new_sentence:
            words = nltk.tokenize.word_tokenize(s)
            final_sentences.append(words)

    model = Word2Vec(
        window=10,
        sg=1,
        hs=0,
        negative=10,
        alpha=0.03,
        min_alpha=0.0007,
        seed=14,
    )

    model.build_vocab(final_sentences, progress_per=200)
    model.train(
        final_sentences, total_examples=model.corpus_count, epochs=100, report_delay=1
    )

    return model


def get_vector(desc, model):

    vec = None

    avgword2vec = None
    count = 0

    for word in desc.split():
        if word in model.wv.key_to_index.keys():
            count += 1
            if avgword2vec is None:
                avgword2vec = model.wv[word]
            else:
                avgword2vec = avgword2vec + model.wv[word]

    if avgword2vec is not None:
        avgword2vec = avgword2vec / count
        return avgword2vec


def get_user_interactions(titles, ratings, original_ids):

    new_idx_df = titles.merge(original_ids, on="title")

    z = zip(new_idx_df.movieId, new_idx_df.index)
    d = dict(z)

    def new_id(id):

        try:
            return d[id]
        except:
            return np.nan

    ratings["movieId"] = ratings["movieId"].apply(new_id)
    ratings = ratings.dropna(subset=["movieId"])
    ratings["movieId"] = ratings["movieId"].astype(int)

    pd.set_option("display.max_columns", None)

    return ratings.groupby("userId")["movieId"].apply(list).to_dict()


def get_user_vectors(user_interactions_dict, movie_vecs):

    user_vecs = {}

    for user in user_interactions_dict:
        user_vec = None
        count = 0

        for film in user_interactions_dict[user]:
            if user_vec is None:
                user_vec = movie_vecs[int(film)]
            else:
                user_vec = user_vec + movie_vecs[int(film)]
            count += 1

        user_vec = user_vec / count
        user_vecs[user] = user_vec

    return user_vecs
