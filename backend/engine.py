import pandas as pd

from backend.utils import extract_movie_id


def movie_recommender_engine(movie_name, matrix, cf_model, n_recs, movie_to_idx, movie_metadata):

    # Extract input movie ID
    movie_id = extract_movie_id(movie_name, movie_metadata)
    # this is a new commit
    movie_idx = movie_to_idx[movie_id]

    # Calculate neighbour distances
    distances, indices = cf_model.kneighbors(matrix[movie_idx], n_neighbors=n_recs)
    movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[
                    :0:-1]

    cf_recs = [
        {'Title': movie_metadata['title'][i[0]], 'Distance': i[1]}
        for i in movie_rec_ids
    ]
    # Select top number of recommendations needed
    df = pd.DataFrame(cf_recs, index=range(1, n_recs))
    print(df)
    return df
