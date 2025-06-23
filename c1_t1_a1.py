import os
import cv2
import numpy as np
import csv
from scipy import signal as sg
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.metrics import accuracy_score
import joblib

# ----- Feature Extraction Functions -----
def laws_texture(gray):
    (rows, cols) = gray.shape[:2]
    smooth_kernel = (1 / 25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray, smooth_kernel, "same")
    gray_processed = np.abs(gray - gray_smooth)

    filter_vectors = np.array([
        [1, 4, 6, 4, 1],    # L5
        [-1, -2, 0, 2, 1],  # E5
        [-1, 0, 2, 0, 1],   # S5
        [1, -4, 6, -4, 1]   # R5
    ])

    filters = [np.matmul(fv1.reshape(5, 1), fv2.reshape(1, 5))
               for fv1 in filter_vectors for fv2 in filter_vectors]

    conv_maps = np.zeros((rows, cols, 16))
    for i in range(16):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    texture_maps = [
        (conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2,
        (conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2,
        (conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2,
        (conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2,
        (conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2,
        (conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2,
        conv_maps[:, :, 10],
        conv_maps[:, :, 5],
        conv_maps[:, :, 15],
        conv_maps[:, :, 0]
    ]

    TEM = [np.abs(texture_maps[i]).sum() / (np.abs(texture_maps[9]).sum() + 1e-6) for i in range(9)]
    return TEM

def lbp_feature(gray):
    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

def glcm_feature(gray):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1, 2]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feats.append(graycoprops(glcm, prop).mean())
    return feats

def gray_histogram(gray):
    hist, _ = np.histogram(gray, bins=128, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

def color_histogram(image):
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist, _ = np.histogram(chan, bins=64, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        features.extend(hist)
    return features

def hog_feature(image):
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd[:128].tolist()

# ----- Load Pretrained Models -----
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
knn = joblib.load('knn_model.pkl')

# ----- Load Query Images -----
query_dir = './query'
query_images = sorted([
    f for f in os.listdir(query_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

query_features = []
query_filenames = []

for fname in query_images:
    img_path = os.path.join(query_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 로드 실패: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feature = (
        laws_texture(gray) +
        lbp_feature(gray) +
        glcm_feature(gray) +
        gray_histogram(gray) +
        color_histogram(img) +
        hog_feature(img)
    )
    query_features.append(feature)
    query_filenames.append(fname)

# ----- 예측 -----
X_scaled = scaler.transform(query_features)
X_pca = pca.transform(X_scaled)
preds = knn.predict(X_pca)

# ----- 결과 저장 -----
with open('c1_t1_a1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for fname, label in zip(query_filenames, preds):
        writer.writerow([fname, label])

print("c1_t1_a1.csv 저장 완료")