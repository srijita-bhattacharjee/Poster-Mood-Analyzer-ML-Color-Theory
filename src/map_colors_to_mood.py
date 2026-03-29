# scripts/map_colors_to_mood.py

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict

# ----------------------------
# Step 1: Load datasets
# ----------------------------

# Poster color features (from extract_colors.py)
poster_df = pd.read_csv('data/poster_colors.csv')

# Color-Pedia dataset
colorpedia_df = pd.read_csv('data/color_pedia.csv')

# Make sure Color-Pedia has RGB columns and emotion/mood column
# Example columns: ['r', 'g', 'b', 'emotion']
# If emotion column is a list of strings, we can take first emotion
colorpedia_df['Emotion'] = colorpedia_df['Emotion'].apply(lambda x: x[0] if isinstance(x, str)==False and isinstance(x, list) else x)

# ----------------------------
# Step 2: Build KDTree for nearest neighbor search
# ----------------------------

color_rgb = colorpedia_df[['R', 'G', 'B']].values
color_mood = colorpedia_df['Emotion'].values

kdtree = KDTree(color_rgb)

# ----------------------------
# Step 3: Function to map a single color to mood
# ----------------------------

def color_to_mood(rgb):
    """
    Find the closest color in Color-Pedia and return its mood
    """
    dist, idx = kdtree.query([rgb], k=1)  # nearest color
    return color_mood[idx[0][0]]

# ----------------------------
# Step 4: Function to map all top colors of a poster
# ----------------------------

def poster_to_mood(row):
    """
    For each poster row, map top 5 colors to moods with weighting
    """
    mood_scores = defaultdict(float)
    
    for i in range(1, 6):  # c1 - c5
        rgb = [row[f'c{i}_R'], row[f'c{i}_G'], row[f'c{i}_B']]
        prop = row[f'c{i}_prop']
        mood = color_to_mood(rgb)
        mood_scores[mood] += prop  # weight by proportion
    
    # Select the mood with the highest score
    final_mood = max(mood_scores, key=mood_scores.get)
    return final_mood

# ----------------------------
# Step 5: Apply function to all posters
# ----------------------------

poster_df['label'] = poster_df.apply(poster_to_mood, axis=1)

# ----------------------------
# Step 6: Save new labeled dataset
# ----------------------------

poster_df.to_csv('data/poster_labeled.csv', index=False)
print("✅ Labeled dataset saved as data/poster_labeled.csv")
