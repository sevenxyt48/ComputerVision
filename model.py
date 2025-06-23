#model.py
import os
import cv2
import numpy as np
from scipy import signal as sg
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ----- Feature Extraction Functions -----
def laws_texture(gray):
    (rows, cols) = gray.shape[:2]
    smooth_kernel = (1/25)*np.ones((5,5))
    gray_smooth = sg.convolve(gray, smooth_kernel, "same")
    gray_processed = np.abs(gray - gray_smooth)
    filter_vectors = np.array([[1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, 1], [1, -4, 6, -4, 1]])
    filters = [np.outer(fv1, fv2) for fv1 in filter_vectors for fv2 in filter_vectors]
    conv_maps = np.zeros((rows, cols, 16))
    for i in range(16):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')
    texture_maps = [(conv_maps[:, :, 1]+conv_maps[:, :, 4])//2, (conv_maps[:, :, 2]+conv_maps[:, :, 8])//2,
                    (conv_maps[:, :, 3]+conv_maps[:, :, 12])//2, (conv_maps[:, :, 7]+conv_maps[:, :, 13])//2,
                    (conv_maps[:, :, 6]+conv_maps[:, :, 9])//2, (conv_maps[:, :, 11]+conv_maps[:, :, 14])//2,
                    conv_maps[:, :, 10], conv_maps[:, :, 5], conv_maps[:, :, 15], conv_maps[:, :, 0]]
    TEM = [np.abs(texture_maps[i]).sum() / (np.abs(texture_maps[9]).sum() + 1e-6) for i in range(9)]
    return TEM

def lbp_feature(gray):
    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
    return (hist.astype("float") / (hist.sum() + 1e-6)).tolist()

def glcm_feature(gray):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1, 2]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return [graycoprops(glcm, prop).mean() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]

def gray_histogram(gray):
    hist, _ = np.histogram(gray, bins=128, range=(0, 256))
    return (hist.astype("float") / (hist.sum() + 1e-6)).tolist()

def color_histogram(image):
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist, _ = np.histogram(chan, bins=64, range=(0, 256))
        features.extend(hist.astype("float") / (hist.sum() + 1e-6))
    return features

def hog_feature(image):
    fd, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd[:128].tolist()

# ----- Training Dataset -----
recaptcha = './Google_Recaptcha_V2_Images_Dataset/images'  # 학습 이미지 경로
labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
          'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']

features = []
targets = []

for label in labels:
    image_dir = os.path.join(recaptcha, label)
    image_list = sorted(os.listdir(image_dir))
    for i, image_name in enumerate(image_list[:80]):  # 클래스당 최대 80장 학습
        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = (laws_texture(gray) + lbp_feature(gray) + glcm_feature(gray) +
             gray_histogram(gray) + color_histogram(img) + hog_feature(img))
        features.append(f)
        targets.append(label)

# ----- 모델 학습 -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine')
knn.fit(X_pca, targets)

# ----- 저장 -----
np.save('X_train.npy', X_pca)
np.save('y_train.npy', np.array(targets))
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(knn, 'knn_model.pkl')

print("✅ 모델 학습 및 저장 완료")
