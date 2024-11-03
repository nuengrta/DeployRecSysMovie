
import streamlit as st
import pickle
from surprise import SVD

# Load data back from the file
with open('66130701910recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit app
st.title("Movie Recommendation System")

# Input for user ID
user_id = st.number_input("Enter your User ID:", min_value=1, max_value=int(movie_ratings['userId'].max()), value=1)

# Get rated movies for the user
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

# Make predictions
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

# Sort predictions by estimated rating in descending order
sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)

# Get top 10 movie recommendations
top_recommendations = sorted_predictions[:10]

# Display recommendations
st.subheader(f"Top 10 Movie Recommendations for User {user_id}:")
for recommendation in top_recommendations:
    movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
    st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")



