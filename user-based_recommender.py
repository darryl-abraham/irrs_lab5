import pandas as pd
import numpy as np

import similarity as sim
import naive_recommender as nav
import utils as ut


def generate_m(movies_idx, users, ratings):
    # Complete the datastructure for rating matrix 
    
    # This function should return a data structure M, such that M[user][movie] yields the rating for a user and a movie.

    m = pd.DataFrame(np.nan, index=users, columns=movies_idx)
    for idx, row in ratings.iterrows():
        m.at[row["userId"], row["movieId"]] = row["rating"]
    
    return m 


def user_based_recommender(target_user_idx, matrix):
    # Compute the similarity between  the target user and each other user in the matrix. 
    # We recomend to store the results in a dataframe (userId and Similarity)

    # Extract the target user's row
    target_user = matrix.loc[target_user_idx]

    similarity_scores = []  # To store (user_id, similarity) tuples

    # Iterate over all other users
    for other_user_id in matrix.index:
        if other_user_id != target_user_idx:  # Skip the target user itself
            # Get the other user's ratings
            other_user = matrix.loc[other_user_id]
            
            # Fill NaN values with 0 for similarity computation
            similarity = sim.compute_similarity(
                target_user.fillna(0).tolist(),
                other_user.fillna(0).tolist()
            )
            
            # Append the similarity score
            similarity_scores.append((other_user_id, similarity))

    u_max = 10 # select u_max most similar users
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[:u_max]
  
    
    # Convert to DataFrame and sort by similarity
    similarity_df = pd.DataFrame(similarity_scores, columns=["userId", "similarity"])
    similarity_df = similarity_df.sort_values(by="similarity", ascending=False)

    print(f"similarity_df.shape = {similarity_df.shape}")

    # Determine the unseen movies by the target user. Those films are identfied 
    # since don't have any rating. 
 
    unseen_movies = target_user[target_user.isna()].index
    print(f"unseen movies: {len(unseen_movies)}")

    # Generate recommendations for unrated movies based on user similarity and ratings.

    recommendations = []

    average_rating_a =  matrix.loc[target_user_idx].mean(skipna=True)

    for movie in unseen_movies:
        sum = 0
        num_of_movies = 0
        for user_idx, similarity in similarity_df.values:
            user_rating = matrix.loc[user_idx, movie] # rating user_idx gave for movie with id "movie"
            average_rating_b = matrix.loc[user_idx].mean(skipna=True)  # avg rating given by user user_idx
            if not np.isnan(user_rating):  # Only consider users who rated this movie
                sum += similarity * (user_rating - average_rating_b)
                num_of_movies += 1
        
        # Calculate predicted rating
        if num_of_movies > 0:
            predicted_rating = average_rating_a + sum
            recommendations.append((movie, predicted_rating))

    # Sort recommendations by predicted rating in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations



if __name__ == "__main__":
    
    # Load the dataset
    path_to_ml_latest_small = './irrs5/data'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)

    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    m = generate_m(movies_idx, users_idy, ratings_train)
        
    # user-to-user similarity
    target_user_idx = 123
    recommendations = user_based_recommender(target_user_idx, m)
     
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    
    # Validation
    matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])
    #print(matrixmpa_genres)
    
     
