import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


chunk_size = 10000
# we are using the small version of the dataset because the original dataset is too large to fit in memory <- this needs to be fixed FAISAL
chunks = pd.read_csv('../input/ratings_small.csv', chunksize=chunk_size)
user_ratings_df = pd.concat(list(chunks), ignore_index=True)
user_ratings_df.head()

movie_metadata = pd.read_csv("../input/movies_metadata.csv", low_memory=False)# movie_metadata = movie_metadata[['title', 'genres']]
movie_metadata.head()
movie_metadata['id'] = pd.to_numeric(movie_metadata['id'], errors='coerce')

# Drop rows with NaN 'id' values
movie_metadata = movie_metadata.dropna(subset=['id'])

# Convert 'id' to int64
movie_metadata['id'] = movie_metadata['id'].astype('int64')
movie_data = user_ratings_df.merge(movie_metadata, left_on='movieId', right_on='id', how='inner')
movie_data.head()

user_ratings_df = user_ratings_df.dropna(subset=['userId', 'movieId'])
user_ratings_df['userId'] = user_ratings_df['userId'].astype('int32')
user_ratings_df['movieId'] = user_ratings_df['movieId'].astype('int32')
print(user_ratings_df.shape)

user_item_matrix = csr_matrix(user_ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0))
