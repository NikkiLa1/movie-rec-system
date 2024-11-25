import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data():
    movies = pd.read_csv(r"C:\Users\psorm\Downloads\movies.csv")
    ratings = pd.read_csv(r"C:\Users\psorm\Downloads\ratings.csv")
    return movies, ratings

@st.cache_data
def preprocess_data(movies, ratings):
    merged_df = pd.merge(movies, ratings, on="movieId", how="inner")
    merged_df['genres'] = merged_df['genres'].astype(str)
    merged_df['genres_split'] = merged_df['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(merged_df['genres_split'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    features = pd.concat([genres_df, merged_df['rating']], axis=1)
    return merged_df, features

@st.cache_resource
def train_knn(features):
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(features.values)  # Ensure it's a NumPy array
    return knn

def recommend(movie_title, knn_model, data, feature_matrix):
    try:
        movie_idx = data[data['title'] == movie_title].index[0]
        distances, indices = knn_model.kneighbors([feature_matrix.iloc[movie_idx].values])
        recommendations = data.iloc[indices[0][1:]]
        return recommendations[['title', 'genres', 'rating']]
    except IndexError:
        return f"Movie '{movie_title}' not found in the dataset."

st.title("Movie Recommendation System")
st.write("Enter a movie title you like, and we will recommend similar ones!")

movies, ratings = load_data()
merged_df, features = preprocess_data(movies, ratings)

knn = train_knn(features)

movie_title = st.text_input("Enter a movie title:", "")

if movie_title:
    st.write(f"Recommendations for: **{movie_title}**")
    recommendations = recommend(movie_title, knn, merged_df, features)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        for _, row in recommendations.iterrows():
            st.write(f"**{row['title']}** - Genres: {row['genres']} - Rating: {row['rating']}")
