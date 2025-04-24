import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords set
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("movies.csv")

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Filter and clean the dataset
required_columns = ["genres", "keywords", "overview", "title"]
df = df[required_columns].dropna().reset_index(drop=True)
df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']
df['cleaned_text'] = df['combined'].apply(preprocess_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the processed data
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
