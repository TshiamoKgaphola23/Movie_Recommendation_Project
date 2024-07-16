import streamlit as st
import pandas as pd
import pickle
from imdb import IMDb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer, cosine similarity matrix, and dataframe
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('cosine_similarity.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# with open('dataframe.pkl', 'rb') as f:
#     v = pickle.load(f)

background_image_url = "https://png.pngtree.com/background/20231030/original/pngtree-3d-render-of-online-ticketing-for-movies-picture-image_5789190.jpg"
background_css = f"""
<style>
    .stApp {{
        background: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

v =  pd.read_csv("ds.csv")

# Initialize IMDbPY
ia = IMDb()

def get_movie_details_from_imdb(movie_title):
    movie_search = ia.search_movie(movie_title)
    if not movie_search:
        return None, 'https://via.placeholder.com/150', 'No description available.'
    
    movie = ia.get_movie(movie_search[0].movieID)
    image_url = movie.get('full-size cover url', 'https://via.placeholder.com/150')
    description = movie.get('plot outline', 'No description available.')
    
    return movie_title, image_url, description

# Define the content-based recommendation function
def get_content_recommendations(title, cosine_sim=cosine_sim):
    if title not in v['title'].values:
        return f"Movie title '{title}' not found in the dataset."

    idx = v[v.title == title].index[0]

    if idx >= len(cosine_sim):
        return f"Index {idx} is out of bounds for cosine_sim matrix."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]   
    movie_indices = [i[0] for i in sim_scores]
    
    return v.iloc[movie_indices]

# Define the recommendation function based on user inputs
def get_recommendations_based_on_criteria(movies_df, genres=None, directors=None, title_casts=None):
    filtered_movies = movies_df
    
    if genres:
        genre_filter = '|'.join(genres)
        filtered_movies = filtered_movies[filtered_movies['content'].str.contains(genre_filter, case=False, na=False)]
    if directors:
        directors_filter = '|'.join(directors)
        filtered_movies = filtered_movies[filtered_movies['content'].str.contains(directors_filter, case=False, na=False)]
    if title_casts:
        filtered_movies = filtered_movies[filtered_movies['content'].str.contains(title_casts, case=False, na=False)]
    
    filtered_movies = filtered_movies.sort_values(by='rating', ascending=False)
    
    return filtered_movies

# Streamlit application
st.title('Movie Recommendation System')

# Prepare the list of movie titles for autocomplete
movie_titles = v['title'].tolist()

# Select recommendation type
recommendation_type = st.selectbox('Choose recommendation type:', ['Content-based', 'Criteria-based'])

if recommendation_type == 'Content-based':
    # Autocomplete movie title input for content-based recommendation
    movie_title = st.multiselect('Enter a movie title:', movie_titles)
    
    # Input genres, directors, and title casts for further filtering
    selected_genres = st.multiselect('Filter by genres:', pd.read_csv("genres.csv")['Genres'].values)
    input_directors = st.multiselect('Filter by directors:', pd.read_csv("directors.csv")['directors'].values)
    input_title_casts = st.multiselect('Filter by title casts :', pd.read_csv("cast.csv")['cast'].values)

    genres = selected_genres if selected_genres else None
    directors = input_directors if input_directors else None
    title_casts = input_title_casts if input_title_casts else None

    # Display selected movies on the left sidebar
    if movie_title:
        st.sidebar.title("Selected Movies")
        for title in movie_title:
            movie_name, image_url, description = get_movie_details_from_imdb(title)
            st.sidebar.image(image_url, width=100)
            st.sidebar.write(f"**{movie_name}**")
            # Add View Details button
            if st.sidebar.button(f"View Details of {movie_name}"):
                st.sidebar.write(description)
            st.sidebar.write("---")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        if movie_title:
            recs = []
            for i, item in enumerate(movie_title):
                recs.append(get_content_recommendations(movie_title[i]))

            if isinstance(recs, str):
                st.write(recs)
            else:
                fr = []
                for i in recs:
                    fr.append(get_recommendations_based_on_criteria(i, genres, directors, title_casts))

                st.write('Content-based Recommendations:')
                for i in fr:
                    if isinstance(i, str):
                        st.write(i)
                    else:
                        for _, row in i.iterrows():
                            title, image_url, description = get_movie_details_from_imdb(row['title'])
                            st.image(image_url, width=100)
                            st.write(f"**{title}**")
                            st.write(description)
                            st.write("---")

elif recommendation_type == 'Criteria-based':
    # # Input genres, directors, and title casts for criteria-based recommendation
    # selected_genres = st.multiselect('Enter genres:', 'genres.csv')
    # input_directors = st.text_input('Enter directors (comma-separated):', '')
    # input_title_casts = st.text_input('Enter title casts (comma-separated):', '')
    selected_genres = st.multiselect('Filter by genres:', pd.read_csv("genres.csv")['Genres'].values)
    input_directors = st.multiselect('Filter by directors:', pd.read_csv("directors.csv")['directors'].values)
    input_title_casts = st.multiselect('Filter by title casts:', pd.read_csv("cast.csv")['cast'].values)


    # Button to get recommendations
    if st.button('Get Recommendations'):
        genres = selected_genres if selected_genres else None
        directors = input_directors if input_directors else None
        title_casts = input_title_casts if input_title_casts else None

        # Get criteria-based recommendations
        recommendations = get_recommendations_based_on_criteria(v, genres, directors, title_casts)

        if not recommendations.empty:
            st.write('Criteria-based Recommendations:')
            for _, row in recommendations.iterrows():
                title, image_url, description = get_movie_details_from_imdb(row['title'])
                st.image(image_url, width=100)
                st.write(f"**{title}** - Rating: {row['rating']}")
                st.write(description)
                st.write("---")
        else:
            st.write('No movies found based on the criteria.')
 




