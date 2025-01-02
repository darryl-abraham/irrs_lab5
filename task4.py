import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import utils as ut
import similarity as sim
import naive_recommender as naive
import user_based_recommender as user

def to_frequencies(vector):
    sum_of_all = sum(vector)
    return vector / sum_of_all

# Load the dataset
path_to_ml_latest_small = './irrs5/data'
dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

# Ratings data
val_movies = 5
ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)


# Create matrix between user and movies 
movies_idx = dataset["movies.csv"]["movieId"]
users_idy = list(set(ratings_train["userId"].values))
m = user.generate_m(movies_idx, users_idy, ratings_train)
    
# Validation
matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])

k = 20

num_of_samples = 15
user_ids = m.index.tolist()  # Get all user IDs (index of the matrix)
random_user_ids = random.sample(user_ids, num_of_samples)  # Randomly select 10 user IDs


# Calculate frequencies for naive recommender
naive_freq = matrixmpa_genres.iloc[0].copy()
naive_freq = naive_freq * 0 # create empty table with all genres

ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
naive_recommendations = naive.naive_recommender(ratings, movies, k, 10)

for movie in naive_recommendations:
    movieId = movie[0]
    naive_freq += matrixmpa_genres.loc[movieId]
naive_freq_vector = to_frequencies(naive_freq.values)

list1 = []
list2 = []

for user_id in random_user_ids:

    # Calculate frequencies for user-user recommender
    user_freq = matrixmpa_genres.iloc[0].copy()
    user_freq = user_freq * 0 # empty

    user_recommendations = [movieId for movieId, rec_rating in user.user_based_recommender(user_id, m)[:k]]
    for movieId in user_recommendations:
        user_freq += matrixmpa_genres.loc[movieId]
    user_freq_vector = to_frequencies(user_freq.values)


    # Calculate frequencies for validation set for this user
    users_ratings = ratings_val[ratings_val['userId'] == user_id] # get all the ratings by the user
    val_movies = users_ratings['movieId'].unique() # get the movies the user has rated

    val_freq = matrixmpa_genres.iloc[0].copy()
    val_freq = val_freq * 0 # empty
    
    for movieId in val_movies:
        val_freq += matrixmpa_genres.loc[movieId]
    val_freq_vector = to_frequencies(val_freq.values)

    # Compute similarities
    sim_naive = sim.compute_similarity(naive_freq_vector, val_freq_vector)
    sim_user = sim.compute_similarity(user_freq_vector, val_freq_vector)

    list1.append(sim_user)
    list2.append(sim_naive)

    print(f"user:{user_id} naive:{sim_naive} user-user:{sim_user}")


# Create a figure and axis
plt.figure(figsize=(8, 4))

# Plot the first list
plt.scatter(list1, [1] * len(list1), color='blue', label='user-user', s=100, alpha=0.5)

# Plot the second list
plt.scatter(list2, [1] * len(list2), color='red', label='naive', s=100, alpha=0.5)

# Add labels and legend
plt.yticks([])  # Remove y-axis ticks for a cleaner look
plt.xlabel('Cosine similarity with the validation vector')
plt.legend()

# Show the plot
plt.show()
    
    
    

