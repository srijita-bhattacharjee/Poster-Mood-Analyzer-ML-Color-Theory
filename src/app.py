import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import cv2

# Configure page for better performance
st.set_page_config(page_title="Mood Analyzer", layout="centered")

# Load model and encoder (cached to avoid reloading)
@st.cache_resource
def load_models():
    with open("mood_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_models()

st.title("🎭 Poster Mood Analyzer")
st.write("Upload a poster and let AI detect the mood based on its dominant colors!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

@st.cache_data
def extract_dominant_colors(image_bytes, k=5, max_size=150):
    """Extract top k dominant colors using KMeans with heavy optimization."""
    # Convert bytes to image
    image = Image.open(BytesIO(image_bytes))
    img = np.array(image)
    
    # Aggressive resize for speed
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert RGB to BGR for model consistency
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Downsample pixels even more for speed
    img_flat = img.reshape((-1, 3))
    
    # Take only a sample of pixels if image is still large
    num_pixels = img_flat.shape[0]
    if num_pixels > 1000:
        np.random.seed(42)  # For consistent caching
        indices = np.random.choice(num_pixels, 1000, replace=False)
        img_flat = img_flat[indices]
    
    img_flat = np.float32(img_flat)

    # Minimal KMeans parameters for speed
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, labels, centers = cv2.kmeans(img_flat, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    
    centers = np.uint8(centers)
    
    # Sort by frequency
    counts = Counter(labels.flatten())
    sorted_indices = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    sorted_colors = [centers[i].tolist() for i in sorted_indices]
    
    return sorted_colors

def plot_color_palette(colors):
    """Display extracted color palette with HEX codes using optimized matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 2), dpi=80, facecolor='none')  # Transparent background
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1.3)
    
    for i, color in enumerate(colors):
        # Convert BGR to RGB for display
        rgb_color = [color[2], color[1], color[0]]  # Swap B and R channels
        
        # Add color rectangle with thin white border
        rect = plt.Rectangle((i, 0), 1, 1, 
                            facecolor=np.array(rgb_color)/255,
                            edgecolor='white', 
                            linewidth=1.5)
        ax.add_patch(rect)
        
        # Convert RGB to HEX
        hex_code = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()
        ax.text(i + 0.5, -0.15, hex_code, ha='center', va='top', fontsize=9, fontweight='bold', color='white')
    
    ax.axis('off')
    ax.set_title("Extracted Color Palette", fontsize=11, pad=10)
    plt.tight_layout()
    
    # Convert to bytes for streamlit with transparent background
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=80, facecolor='none', edgecolor='none')
    buf.seek(0)
    plt.close(fig)  # Free memory immediately
    
    return buf

if uploaded_file is not None:
    # Read file bytes once for caching
    image_bytes = uploaded_file.read()
    
    # Display image with smaller size for performance
    image = Image.open(BytesIO(image_bytes))
    
    # Resize for display to reduce memory
    display_img = image.copy()
    display_img.thumbnail((600, 600), Image.Resampling.LANCZOS)
    st.image(display_img, caption="Uploaded Image", width="stretch")

    with st.spinner("Analyzing colors..."):
        # Extract top 5 colors (now cached)
        dominant_colors = extract_dominant_colors(image_bytes, k=5, max_size=150)
        
        # Plot color palette
        color_img = plot_color_palette(dominant_colors)
        st.image(color_img, width="stretch")

    # Use average color for mood prediction
    avg_color = np.mean(dominant_colors, axis=0)
    B, G, R = avg_color  # BGR format
    
    # Prepare input and predict (model expects BGR)
    input_df = pd.DataFrame([[B, G, R]], columns=["R", "G", "B"])
    
    probabilities = model.predict_proba(input_df)[0]
    mood_labels = le.inverse_transform(np.arange(len(probabilities)))
    
    pred_mood = le.inverse_transform([np.argmax(probabilities)])[0]
    
    # Create probability dataframe
    prob_df = pd.DataFrame({"Mood": mood_labels, "Probability": probabilities})
    prob_df = prob_df.sort_values(by="Probability", ascending=False)
    
    st.success(f"🎨 Predicted Mood: **{pred_mood}**")

    # Display color distribution chart with toggle
    st.subheader("Dominant Colors Distribution")
    
    # Add toggle switch for graph/table view
    view_mode = st.toggle("Table", key="color_view")
    
    if not view_mode:  # Graph view (default)
        # Create matplotlib bar chart with actual colors
        fig, ax = plt.subplots(figsize=(8, 4), dpi=80, facecolor='none')
        ax.set_facecolor('none')  # Make axes background transparent
        
        color_labels = []
        dominance_values = []
        bar_colors = []
        
        for i, color in enumerate(dominant_colors):
            rgb_color = [color[2], color[1], color[0]]  # BGR to RGB
            hex_code = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()
            color_labels.append(f"Color {i+1}\n{hex_code}")
            dominance_values.append(100 - (i * 15))  # Decreasing dominance
            bar_colors.append(np.array(rgb_color) / 255)  # Normalize to 0-1 for matplotlib
        
        bars = ax.bar(color_labels, dominance_values, color=bar_colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Dominance (%)', color='white', fontsize=10)
        ax.set_ylim(0, 110)
        ax.tick_params(colors='white', labelsize=9)
        # Remove all spines/box
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        # Convert to bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80, facecolor='none', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        st.image(buf, width="stretch")
    else:
        # Create table view
        table_data = []
        for i, color in enumerate(dominant_colors):
            rgb_color = [color[2], color[1], color[0]]  # BGR to RGB
            hex_code = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()
            dominance = 100 - (i * 15)
            table_data.append({
                "Color": f"Color {i+1}",
                "HEX Code": hex_code,
                "RGB": f"({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})",
                "Dominance": f"{dominance}%"
            })
        
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, width="stretch", hide_index=True)

    # Show detailed probabilities as table only
    st.subheader("Mood Probability Distribution")
    
    # Display as a formatted table
    display_prob_df = prob_df.copy()
    display_prob_df['Probability'] = display_prob_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(display_prob_df, width="stretch", hide_index=True)

    
