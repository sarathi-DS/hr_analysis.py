import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

anime_file = pd.read_csv('anime_recommendation_dataset.csv')
print(anime_file)
print(anime_file.columns)
print('--------------------------------------------')
#finding null data
print(anime_file.isna().sum())
# drop null values
anime_file_cleaned = anime_file.dropna(inplace=True)
# print(f"Original number of rows: {len(anime_file)}")

#find the data types
print(anime_file.dtypes)
def clean_data(x):
    if isinstance(x,str):
        return str.lower((x.replace(',',"")))
    return ''
# Apply cleaning to the text columns
for col in ['genres', 'characters', 'title', 'synopsis']:
    anime_file[col] = anime_file[col].apply(clean_data)
# Create the 'soup' column by combining the relevant features
# The 'title' and 'synopsis' are generally less useful than 'genres' and 'characters'
# But let's include all for a rich description
def create_soup(x):
    return ' '.join([x['genres'], x['characters'], x['title'], x['synopsis']])

anime_file['soup'] = anime_file.apply(create_soup, axis=1)
print(anime_file['soup'])
print("\n### Sample of the new 'soup' column (the content description) ###")
print(anime_file['soup'].head(1).iloc[0][:150] + "...")
print("### Shape of the Cosine Similarity Matrix (Items x Items) ###")
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object to transform the text into a matrix of token counts
# 'stop_words="english"' removes common words like 'a', 'the', 'is'
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(anime_file['soup'])
print(count_matrix)
print(count.get_feature_names_out()[733])

# The result is a sparse matrix, which is a numerical representation of your content
print("\n### Shape of the Count Matrix (Items x Features) ###")
print(count_matrix.shape)
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity matrix
# This takes the count_matrix and calculates the similarity between every pair of items.
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Print the shape of the similarity matrix
# If you have 'N' items, the result will be an N x N matrix.
# Each cell (i, j) holds the similarity score between item i and item j.
print("### Shape of the Cosine Similarity Matrix (Items x Items) ###")
print(cosine_sim.shape)

# Let's verify the diagonal (an item is 100% similar to itself)
# print(cosine_sim[0][0])
# Create a reverse map for quick title lookup
# This creates a Series where the index is the title and the value is the DataFrame index (row number)
indices = pd.Series(anime_file.index, index=anime_file['title']).drop_duplicates()
print('-------',indices)
def get_recommendations(title,cosine_sim,anime_file,indices=indices):
    # 1. Get the index (row number) of the movie that matches the title
    # We use .get(title) to safely retrieve the index
    idx = indices.get(title)
    if idx is None:
        return "Item not found in the database. Check the spelling or item list."
    sim_scores = list(enumerate(cosine_sim[idx]))
    print(sim_scores)
    # --- The rest of your function code will go here, indented as well ---
    # 2. Sort the movies based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 3. Get the scores of the Top 11 most similar movies (excluding the item itself)
    sim_scores = sim_scores[1:11]
    # 4. Get the movie indices and titles
    item_indices = [i[0] for i in sim_scores]
    recommendations = anime_file['title'].iloc[item_indices]

    return recommendations
    # 5. Get the item indices and titles
    item_indices = [i[0] for i in sim_scores]
    recommendations = df['title'].iloc[item_indices]

# To find a title to test, look at the titles in your cleaned DataFrame:
# print(df_cleaned['title'].head(5))

# Pick a title (make sure it's lowercased and matches your clean data format)
# test_title = 'cowboy bebop' # E.g., 'naruto' or 'toy story'

# if test_title in indices:
#     print(f"\n###  Recommendations for someone who liked '{test_title}'  ###")
#     results = get_recommendations(test_title)
#     print(results)
# else:
#     print(f"The test title '{test_title}' was not found in the index. Please verify the title spelling and case.")
# 1. Define the title (ensure it's lowercase, as verified previously)
test_title = 'cowboy bebop'

# 2. Call the function, providing the two missing arguments:
results = get_recommendations(test_title, cosine_sim, anime_file)

print(f"\n###  Recommendations for someone who liked '{test_title}'  ###")
print(results)