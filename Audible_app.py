import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Page configuration
st.set_page_config(
    page_title="Audible Book Recommendations",
    page_icon="📚",
    layout="wide"
)

# Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Audible_Catalog.csv")
    
    # FIX: Convert Genres from string back to list if needed
    def safe_eval_list(x):
        if pd.isna(x):
            return ['NONE']
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                # Try to evaluate as Python literal
                result = ast.literal_eval(x)
                if isinstance(result, list):
                    return result
                return [str(result)]
            except:
                # If it fails, treat as single genre
                return [x] if x != 'NONE' else ['NONE']
        return ['NONE']
    
    df['Genres'] = df['Genres'].apply(safe_eval_list)
    
    # Recreate Genres_Text if needed
    df['Genres_Text'] = df['Genres'].apply(lambda x: ' | '.join(x) if isinstance(x, list) else str(x))
    
    return df

# Load data
try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded {len(df)} books")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please make sure 'Cleaned_Audible_Catalog.csv' exists in C:/Users/vidhy/Downloads/audible-book-recommendation/")
    st.stop()

# Get all unique genres (excluding NONE)
@st.cache_data
def get_all_genres():
    all_genres = set()
    for genres_list in df['Genres']:
        if isinstance(genres_list, list):
            all_genres.update(genres_list)
    # Remove NONE and sort
    all_genres.discard('NONE')
    return sorted(list(all_genres))

all_genres = get_all_genres()

# TF-IDF for content-based similarity
@st.cache_resource
def build_recommendation_model():
    df_copy = df.copy()
    df_copy['Combined_Features'] = (
        df_copy['Description'].fillna('') + ' ' + 
        df_copy['Genres_Text'].fillna('')
    )
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df_copy['Combined_Features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df_copy.index, index=df_copy['Book Name']).drop_duplicates()
    
    return cosine_sim, indices

cosine_sim, indices = build_recommendation_model()

# Helper function to find book
def find_book_title(partial_title):
    if partial_title in indices:
        return partial_title
    matches = df[df['Book Name'].str.contains(partial_title, case=False, na=False)]['Book Name']
    if len(matches) == 1:
        return matches.iloc[0]
    elif len(matches) > 1:
        return matches.iloc[0]
    return None

# Recommend books by content similarity
def recommend_books(title, top_n=5):
    actual_title = find_book_title(title)
    if actual_title is None:
        return pd.DataFrame({'Message': [f"'{title}' not found."]})
    
    idx = indices.get(actual_title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    
    results = df.iloc[book_indices][['Book Name', 'Author', 'Primary_Genre', 'Rating']].copy()
    results['Similarity'] = [round(score[1], 3) for score in sim_scores]
    return results

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📚 Audible Book Recommendation App")
st.markdown("*Discover your next favorite audiobook with AI-powered recommendations*")

# Sidebar navigation
option = st.sidebar.radio(
    "Navigation",
    ['📖 Recommend by Book', '🎯 Genre-Based Book Finder', '🎨 Multi-Genre Search', '📊 EDA']
)

# ============================================================================
# Page 1: Recommend by Book
# ============================================================================
if option == '📖 Recommend by Book':
    st.header("🔍 Find Similar Books")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        book_title = st.selectbox(
            "Choose a book you like",
            sorted(df['Book Name'].unique()),
            help="Select a book to get similar recommendations"
        )
    
    with col2:
        top_n = st.slider("Number of recommendations", 1, 20, 5)
    
    # Show selected book's info
    if book_title:
        selected_book = df[df['Book Name'] == book_title].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rating", f"⭐ {selected_book['Rating']}")
        col2.metric("Reviews", f"{int(selected_book['Number of Reviews'])}")
        col3.metric("Length", f"{int(selected_book['Listening Time (min)'])} min")
        
        if isinstance(selected_book['Genres'], list) and selected_book['Genres'][0] != 'NONE':
            st.info(f"**Genres:** {', '.join(selected_book['Genres'])}")
    
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Finding similar books..."):
            results = recommend_books(book_title, top_n)
            
            if 'Message' in results.columns:
                st.error(results['Message'].iloc[0])
            else:
                st.success(f"Found {len(results)} similar books!")
                st.dataframe(results, width='stretch', hide_index=True)

# ============================================================================
# Page 2: Genre-Based Book Finder
# ============================================================================
elif option == '🎯 Genre-Based Book Finder':
    st.header("🎧 Genre-Based Book Finder")
    
    if not all_genres:
        st.error("❌ No genres found in the dataset!")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        genre = st.selectbox("Choose a genre", all_genres)
    
    with col2:
        top_n = st.slider("Number of books to show", 1, 20, 10)
    
    with col3:
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 4.0, 0.5)
    
    # Filter books by genre
    mask = df['Genres'].apply(lambda x: genre in x if isinstance(x, list) else False)
    filtered = df[mask & (df['Rating'] >= min_rating)].sort_values(
        by=['Rating', 'Number of Reviews'], 
        ascending=[False, False]
    ).head(top_n)
    
    st.subheader(f"📚 Top {len(filtered)} books in {genre}")
    
    if len(filtered) == 0:
        st.warning(f"No books found in '{genre}' with rating ≥ {min_rating}")
    else:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Books Found", len(filtered))
        col2.metric("Avg Rating", f"{filtered['Rating'].mean():.2f} ⭐")
        col3.metric("Avg Length", f"{filtered['Listening Time (min)'].mean():.0f} min")
        
        st.dataframe(
            filtered[['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Listening Time (min)']],
            width='stretch',
            hide_index=True
        )

# ============================================================================
# Page 3: Multi-Genre Search
# ============================================================================
elif option == '🎨 Multi-Genre Search':
    st.header("🎨 Find Books by Multiple Genres")
    
    search_type = st.radio(
        "Search type",
        ['Books with ANY of these genres', 'Books with ALL of these genres']
    )
    
    selected_genres = st.multiselect(
        "Select genres (up to 5)",
        all_genres,
        max_selections=5,
        help="Choose multiple genres to find books"
    )
    
    if selected_genres:
        if search_type == 'Books with ANY of these genres':
            mask = df['Genres'].apply(
                lambda x: any(g in x for g in selected_genres) if isinstance(x, list) else False
            )
        else:
            mask = df['Genres'].apply(
                lambda x: all(g in x for g in selected_genres) if isinstance(x, list) else False
            )
        
        filtered = df[mask].sort_values(
            by=['Rating', 'Number of Reviews'],
            ascending=[False, False]
        ).head(20)
        
        st.subheader(f"Found {len(filtered)} books")
        
        if len(filtered) == 0:
            st.warning("No books found with the selected genre combination")
        else:
            st.dataframe(
                filtered[['Book Name', 'Author', 'Genres', 'Rating', 'Number of Reviews']],
                width='stretch',
                hide_index=True
            )
    else:
        st.info("👆 Please select at least one genre to start searching")

# ============================================================================
# Page 4: EDA
# ============================================================================
elif option == '📊 EDA':
    st.header("📊 Exploratory Data Analysis")
    
    # Dataset statistics
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Books", len(df))
    col2.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    col3.metric("Unique Genres", len(all_genres))
    col4.metric("Books with Genres", len(df[df['Primary_Genre'] != 'NONE']))
    
    # Ratings Distribution
    st.subheader("⭐ Distribution of Ratings")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Rating'], bins=10, kde=True, ax=ax1, color="skyblue")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)
    
    # Top Genres
    st.subheader("🎯 Top 15 Primary Genres")
    top_genres = df[df['Primary_Genre'] != 'NONE']['Primary_Genre'].value_counts().head(15)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, ax=ax2, palette="viridis", legend=False)
    ax2.set_xlabel("Number of Books")
    ax2.set_ylabel("Genre")
    st.pyplot(fig2)
    
    # Genre Count Distribution
    st.subheader("🔗 Genres per Book")
    df['Genre_Count'] = df['Genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Genre_Count'], bins=range(1, df['Genre_Count'].max()+2), ax=ax3, color="coral")
    ax3.set_xlabel("Number of Genres per Book")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use partial book titles for faster search!")