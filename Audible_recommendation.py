###Data Preparation:

import pandas as pd
import numpy as np
import re

# Load the datasets
df1 = pd.read_csv("Audible_Catlog.csv")
df2 = pd.read_csv("Audible_Catlog_Advanced_Features.csv")

# Merge datasets on 'Book Name' and 'Author'
merged_df = pd.merge(df1, df2, on=["Book Name", "Author"], suffixes=('_x', ''))

# Drop duplicate columns after merge
merged_df.drop(columns=["Rating_x", "Number of Reviews_x", "Price_x"], inplace=True)

# Rename retained columns for clarity
merged_df.rename(columns={
    "Rating": "Rating",
    "Number of Reviews": "Number of Reviews",
    "Price": "Price"
}, inplace=True)

# Remove duplicates based on Book Name and Author
merged_df.drop_duplicates(subset=["Book Name", "Author"], inplace=True)

# Drop rows with missing critical information
merged_df.dropna(subset=["Rating", "Number of Reviews", "Price", "Listening Time"], inplace=True)

# Convert 'Listening Time' into total minutes
def convert_time_to_minutes(time_str):
    match = re.match(r'(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*minutes?)?', time_str.strip())
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return np.nan

merged_df['Listening Time (min)'] = merged_df['Listening Time'].apply(convert_time_to_minutes)

# Extract genre from 'Ranks and Genre'
def extract_all_genres(text):
    """Extract all genres from the Ranks and Genre column"""
    if pd.isna(text) or text == '-1' or text.strip() == '':
        return ['NONE']
    
    genres = []
    rankings = text.split(',')
    
    for rank in rankings:
        match = re.search(r'#\d+\s+(?:Free\s+)?in\s+(.+?)(?:\s*\(|$)', rank.strip())
        if match:
            genre = match.group(1).strip()
            # Skip generic category and avoid duplicates
            if genre != "Audible Audiobooks & Originals" and genre not in genres:
                genres.append(genre)
    
    return genres if genres else ['NONE']


# Apply to your merged dataframe (replace old Genre extraction):
merged_df['Genres'] = merged_df['Ranks and Genre'].apply(extract_all_genres)
merged_df['Primary_Genre'] = merged_df['Genres'].apply(lambda x: x[0] if x and len(x) > 0 else 'NONE')
merged_df['Genres_Text'] = merged_df['Genres'].apply(lambda x: ' | '.join(x) if x else 'NONE')

merged_df.to_csv("Cleaned_Audible_Catalog.csv", index=False)

# Preview cleaned data
print("✅ Genre extraction complete!")
print("\nSample results:")
print(merged_df[['Book Name', 'Primary_Genre', 'Genres']].head())
print(f"\nTotal unique genres found: {len(set([g for genres in merged_df['Genres'].dropna() for g in genres]))}")

### EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df = pd.read_csv("Cleaned_Audible_Catalog.csv")

# Set up visualization styles
sns.set(style="whitegrid")
plt.figure(figsize=(8, 3))

# 1. Ratings Distribution
sns.histplot(df['Rating'], bins=10, kde=True, color="skyblue")
plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# 2. Top Genres by Count
top_genres = df[df['Primary_Genre'] != 'NONE']['Primary_Genre'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette="viridis", legend=False)
plt.title("Top 10 Most Common Primary Genres")
plt.xlabel("Number of Books")
plt.ylabel("Genre")
plt.show()

# 3. Top-Rated Books (Rating ≥ 4.8)
top_books = df[(df['Rating'] >= 4.8) & (df['Primary_Genre'] != 'NONE')][
    ['Book Name', 'Author', 'Rating', 'Primary_Genre']
].drop_duplicates()
print("\nTop-Rated Books (Rating ≥ 4.8):")
print(top_books.head(20))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load cleaned dataset
df = pd.read_csv("Cleaned_Audible_Catalog.csv")

# Combine 'Book Name', 'Description', and 'Genres' for better NLP features
df['Text'] = (
    df['Book Name'].fillna('') + " " + 
    df['Description'].fillna('') + " " +
    df['Genres_Text'].fillna('')
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = tfidf.fit_transform(df['Text'])

# === ELBOW METHOD to find optimal K ===
print("Finding optimal number of clusters...")
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_tfidf)
    inertias.append(kmeans_temp.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# === K-MEANS CLUSTERING ===
k = 10  # You can adjust based on elbow plot
print(f"\nPerforming K-Means clustering with k={k}...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_tfidf)

# === PCA for VISUALIZATION ===
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Enhanced scatter plot
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=df['Cluster'], 
    palette="Set2",
    s=50,
    alpha=0.6,
    edgecolor='black',
    linewidth=0.5
)
plt.title("K-Means Clustering of Books Based on Title, Description & Genres", fontsize=14)
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# === CLUSTER ANALYSIS ===
print("\n" + "="*70)
print("CLUSTER ANALYSIS")
print("="*70)

for i in range(k):
    cluster_books = df[df['Cluster'] == i]
    
    print(f"\n📚 CLUSTER {i} ({len(cluster_books)} books)")
    print("-" * 70)
    
    # Top genres in this cluster
    top_genres = cluster_books[cluster_books['Primary_Genre'] != 'NONE']['Primary_Genre'].value_counts().head(3)
    print(f"🎯 Top Genres: {', '.join([f'{g} ({c})' for g, c in top_genres.items()])}")
    
    # Average rating
    avg_rating = cluster_books['Rating'].mean()
    print(f"⭐ Average Rating: {avg_rating:.2f}")
    
    # Average listening time
    avg_time = cluster_books['Listening Time (min)'].mean()
    print(f"⏱️  Average Listening Time: {avg_time:.0f} minutes ({avg_time/60:.1f} hours)")
    
    # Example book titles
    print(f"\n📖 Example Titles:")
    sample_titles = cluster_books.nlargest(5, 'Rating')[['Book Name', 'Author', 'Rating']].head(5)
    for idx, row in sample_titles.iterrows():
        print(f"   • {row['Book Name']} by {row['Author']} ({row['Rating']}⭐)")

# === CLUSTER SIZE DISTRIBUTION ===
plt.figure(figsize=(8, 5))
cluster_sizes = df['Cluster'].value_counts().sort_index()
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, hue=cluster_sizes.index, palette="Set2", legend=False)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.title('Distribution of Books Across Clusters', fontsize=14)
for i, v in enumerate(cluster_sizes.values):
    plt.text(i, v + 5, str(v), ha='center', fontsize=10)
plt.show()

# === GENRE DISTRIBUTION BY CLUSTER ===
# Create a heatmap showing genre distribution across clusters
from collections import Counter

cluster_genre_matrix = []
all_top_genres = df[df['Primary_Genre'] != 'NONE']['Primary_Genre'].value_counts().head(10).index

for i in range(k):
    cluster_books = df[df['Cluster'] == i]
    genre_counts = cluster_books['Primary_Genre'].value_counts()
    cluster_genre_matrix.append([genre_counts.get(g, 0) for g in all_top_genres])

plt.figure(figsize=(12, 6))
sns.heatmap(
    np.array(cluster_genre_matrix).T, 
    annot=True, 
    fmt='d', 
    cmap='YlOrRd',
    xticklabels=[f'Cluster {i}' for i in range(k)],
    yticklabels=all_top_genres,
    cbar_kws={'label': 'Number of Books'}
)
plt.title('Genre Distribution Across Clusters', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.tight_layout()
plt.show()

print("\n✅ Clustering analysis complete!")