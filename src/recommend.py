import joblib

# Load data
df = joblib.load('df_cleaned.pkl')
cosine_sim = joblib.load('cosine_sim.pkl')

def recommend_movies(movie_name, top_n=5):
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."

    return result_df
