import pandas as pd
import numpy as np
import pickle
from utils import get_desc, build_model, get_vector, get_user_vectors
import warnings; warnings.simplefilter('ignore')

metadata = pd. read_csv('./original_data/movies_metadata.csv')
metadata['production_companies'] = metadata['production_companies'].fillna('[]')
metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

credits = pd.read_csv('./original_data/credits.csv')
metadata = metadata[pd.to_numeric(metadata['id'], errors='coerce').notnull()]
metadata['id'] = metadata['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata = metadata.merge(credits, on='id')
metadata = get_desc(metadata)
model = build_model(metadata)
metadata['vec'] = metadata['desc'].apply(get_vector, args=(model,)).copy()
metadata = metadata[metadata['vec'].notna()]
metadata = metadata.reset_index()
vec_matrix = metadata['vec'].tolist()

with open("./cleaned_data/user_interacted_items.pkl","rb") as f:
    user_interacted_items = pickle.load(f)

user_vectors = get_user_vectors(user_interacted_items, vec_matrix)

with open('user_vectors.pkl', 'wb') as f:
    pickle.dump(user_vectors, f)

with open('movie_vectors.pkl', 'wb') as f:
    pickle.dump(vec_matrix, f)


print("build complete")