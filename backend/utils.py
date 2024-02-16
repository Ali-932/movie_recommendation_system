def extract_movie_id(movie_name, movie_df):
    return movie_df.loc[movie_df['title'] == movie_name, 'id'].values[0]


def extract_movie_name(movie_id, movie_df):
    return movie_df.loc[movie_df['id'] == movie_id, 'title'].values[0]
