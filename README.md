# 📚 Audible Book Recommendation System

An AI-powered book recommendation system that uses Natural Language Processing (NLP) and Machine Learning to help users discover their next favorite audiobook from the Audible catalog.

## 🌟 Features

- **📖 Content-Based Recommendations**: Get personalized book suggestions based on book descriptions and genres using TF-IDF and cosine similarity
- **🎯 Genre-Based Filtering**: Discover top-rated books in specific genres
- **🎨 Multi-Genre Search**: Find books that match multiple genre preferences
- **📊 Interactive Data Visualization**: Explore dataset insights with dynamic charts
- **🔍 Fuzzy Search**: Find books using partial titles or keywords
- **⚡ Real-Time Results**: Get instant recommendations powered by scikit-learn

## 🏗️ Architecture

```
┌─────────────────┐
│  User Interface │ (Streamlit)
└────────┬────────┘
         │
┌────────▼────────┐
│ Recommendation  │ (TF-IDF + Cosine Similarity)
│     Engine      │
└────────┬────────┘
         │
┌────────▼────────┐
│  Data Processing│ (Pandas + NumPy)
└────────┬────────┘
         │
┌────────▼────────┐
│   Audible Data  │ (CSV Files)
└─────────────────┘
```

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **scikit-learn** | Machine learning (TF-IDF, Cosine Similarity, K-Means) |
| **Matplotlib & Seaborn** | Data visualization |
| **AWS EC2** | Cloud deployment |

## 📊 How It Works

### 1. Data Processing
- Merges multiple Audible catalog datasets
- Extracts and processes book metadata (title, author, genre, rating, etc.)
- Cleans and transforms listening time into standardized format
- Handles multiple genres per book

### 2. Feature Engineering
- **TF-IDF Vectorization**: Converts book descriptions and genres into numerical features
- **Genre Extraction**: Uses regex to parse and categorize books into multiple genres
- **Text Processing**: Combines book titles, descriptions, and genres for comprehensive analysis

### 3. Recommendation Algorithm
- **Content-Based Filtering**: Uses cosine similarity to find books with similar content
- **Genre Matching**: Filters books by single or multiple genre preferences
- **Weighted Scoring**: Considers ratings and review counts for quality recommendations

### 4. Clustering Analysis
- **K-Means Clustering**: Groups similar books together
- **PCA Visualization**: Reduces dimensionality for 2D visualization
- **Cluster Insights**: Identifies patterns in book collections

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhakshna1104/Audible-book-recommendation.git
   cd Audible-book-recommendation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run audible_app.py
   ```

5. **Open in browser**
   ```
   http://localhost:8501
   ```

## 📁 Project Structure

```
Audible-book-recommendation/
│
├── audible_app.py                          # Main Streamlit application
├── Audible_recommendation.py               # Data processing and analysis
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
│
├── Audible_Catlog.csv                     # Original dataset 1
├── Audible_Catlog_Advanced_Features.csv   # Original dataset 2
├── Cleaned_Audible_Catalog.csv            # Processed dataset
│
└── .gitignore                              # Git ignore rules
```

## 📖 Usage Guide

### 1. Find Similar Books

```python
# Navigate to "Recommend by Book" page
# Select a book from the dropdown
# Get 5-20 similar book recommendations
```

### 2. Browse by Genre

```python
# Navigate to "Genre-Based Book Finder"
# Select a genre (e.g., "Personal Success")
# Adjust minimum rating threshold
# View top-rated books in that genre
```

### 3. Multi-Genre Search

```python
# Navigate to "Multi-Genre Search"
# Select multiple genres
# Choose "ANY" or "ALL" matching
# Discover books across genres
```

### 4. Explore Data

```python
# Navigate to "EDA" page
# View rating distributions
# See top genres
# Analyze genre combinations
```

## 🎯 Features Breakdown

### Content-Based Recommendations

Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to analyze:
- Book titles
- Descriptions
- Genre tags

Computes **cosine similarity** to find books with similar content profiles.

### Genre Extraction

Smart regex-based extraction that:
- Parses ranking data from Audible
- Extracts multiple genres per book
- Filters out generic categories
- Handles edge cases (missing data, special characters)

### Clustering Analysis

**K-Means Algorithm** for:
- Grouping similar books
- Discovering hidden patterns
- Visualizing book relationships
- Genre distribution analysis

## 📊 Dataset Information

### Source
- Audible audiobook catalog
- 10,000+ books
- Multiple genres and categories

### Features
- **Book Name**: Title of the audiobook
- **Author**: Book author(s)
- **Rating**: Average user rating (1-5)
- **Number of Reviews**: Total review count
- **Price**: Book price (in currency)
- **Listening Time**: Duration in hours and minutes
- **Description**: Book synopsis
- **Genres**: Multiple genre classifications
- **Ranks**: Category rankings on Audible

## 🚀 Deployment

### Deploy on AWS EC2

1. **Launch EC2 Instance**
   - Ubuntu 22.04 LTS
   - t2.micro (Free tier)
   - Open ports: 22, 80, 8501

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ec2-user@your-ec2-ip
   
   # Install dependencies
   sudo apt update && sudo apt install python3-pip python3-venv git -y
   
   # Clone and setup
   git clone https://github.com/dhakshna1104/Audible-book-recommendation.git
   cd Audible-book-recommendation
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Run app
   streamlit run audible_app.py
   ```

3. **Access**
   ```
   http://YOUR_EC2_IP:8501
   ```
