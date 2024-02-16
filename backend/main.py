from backend.engine import movie_recommender_engine
from backend.model_training import model_learning
from input_cleaning import get_input_and_clean

if __name__ == "__main__":
    user_ratings_df, movie_metadata, user_item_matrix = get_input_and_clean()

    movie_to_idx = {
        movie_id: i for i, movie_id in enumerate(user_ratings_df['movieId'].unique())
    }

    cf_knn_model = model_learning(user_item_matrix)

    movie_recommender_engine('Rocky III', user_item_matrix, cf_knn_model, 10, movie_to_idx, cf_knn_model, movie_metadata)
