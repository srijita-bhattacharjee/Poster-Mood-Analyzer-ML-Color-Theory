import pandas as pd
import webcolors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# ---- Load dataset ----
poster_df = pd.read_csv('C:/Users/sriji/aiml_proj/data/color_pedia.csv')  # <-- replace with your actual CSV filename
print(f"Poster dataset loaded: {poster_df.shape}")

# ---- Convert color names to RGB ----
def color_to_rgb(name):
    try:
        return webcolors.name_to_rgb(name)
    except ValueError:
        return (0, 0, 0)  # fallback for unknown colors

poster_df[['R', 'G', 'B']] = poster_df['Color Name'].apply(lambda x: pd.Series(color_to_rgb(x)))

# ---- Encode Mood labels ----
label_encoder = LabelEncoder()
poster_df['Mood_encoded'] = label_encoder.fit_transform(poster_df['Mood'])
print(f"Encoded Moods: {list(label_encoder.classes_)}")

# ---- Features and target ----
X = poster_df[['R', 'G', 'B']]
y = poster_df['Mood_encoded']

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train model ----
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model trained successfully!")

# ---- Save model & label encoder ----
with open('mood_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved!")
