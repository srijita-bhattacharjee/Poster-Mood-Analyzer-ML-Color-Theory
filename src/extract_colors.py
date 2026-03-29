import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv
from skimage import io

def get_dominant_colors(img_path, k=5, resize=600):
    img = io.imread(img_path)
    # Resize for speed
    h, w = img.shape[:2]
    scale = resize / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(float)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    centers = np.rint(kmeans.cluster_centers_).astype(int)
    counts = np.bincount(kmeans.labels_)
    idx = np.argsort(-counts)
    centers = centers[idx]
    counts = counts[idx] / counts.sum()
    return centers, counts

def rgb_to_hsv_normalized(r, g, b):
    r_, g_, b_ = r/255.0, g/255.0, b/255.0
    h, s, v = rgb_to_hsv(r_, g_, b_)
    return h, s, v

def process_folder(folder, out_csv='features.csv', k=5):
    rows = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        path = os.path.join(folder, fname)
        centers, counts = get_dominant_colors(path, k=k)
        dom_r, dom_g, dom_b = centers[0]
        avg_h = np.mean([rgb_to_hsv_normalized(*c)[0] for c in centers])
        avg_s = np.mean([rgb_to_hsv_normalized(*c)[1] for c in centers])
        avg_v = np.mean([rgb_to_hsv_normalized(*c)[2] for c in centers])
        row = {
            'filename': fname,
            'dom_R': int(dom_r),
            'dom_G': int(dom_g),
            'dom_B': int(dom_b),
            'avg_H': float(avg_h),
            'avg_S': float(avg_s),
            'avg_V': float(avg_v)
        }
        for i, (c, p) in enumerate(zip(centers, counts), 1):
            row[f'c{i}_R'], row[f'c{i}_G'], row[f'c{i}_B'] = int(c[0]), int(c[1]), int(c[2])
            row[f'c{i}_prop'] = float(p)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__ == "__main__":
    process_folder('data/posters', out_csv='data/poster_colors.csv', k=5)
