import pandas as pd 
import utils as ut

def naive_recommender(ratings: object, movies:object, k: int = 10, t: int = 0) -> list: 
    # Provide the code for the naive recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.

    # Calculate the average rating and the number of ratings for each movie
    movie_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()

    # Filter movies with at least `t` ratings
    movie_stats = movie_stats[movie_stats["num_ratings"] > t]

    # Merge the statistics with movie titles
    movies_with_ratings = movies.merge(movie_stats, on="movieId")

    # Sort movies by average rating in descending order
    top_movies = movies_with_ratings.sort_values("avg_rating", ascending=False)

    # Get the top `k` movies with titles, average ratings, and number of ratings as tuples
    most_seen_movies = list(zip(top_movies["title"], top_movies["avg_rating"].round(3), top_movies["num_ratings"]))

    return most_seen_movies[:k]


if __name__ == "__main__":
    
    path_to_ml_latest_small = './data'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    recs = naive_recommender(ratings, movies, 10, 10)
    print(recs)

