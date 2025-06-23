import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os
from PIL import Image
from torch.utils.data import Dataset

# ------------------------
# 환경 설정
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
query_dir = "./query"
class_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
               'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.filenames = [os.path.basename(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.filenames[idx]


# ------------------------
# 전처리 설정
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------
# 데이터셋 로딩 및 분할
# ------------------------
query_dataset = CustomImageDataset(query_dir, transform=transform)
query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])  # FC 제거
model.to(device)
model.eval()

# ------------------------
# Feature 추출 함수
# ------------------------
def extract_features(dataloader):
    features = []
    filenames = []
    with torch.no_grad():
        for x, names in tqdm(dataloader):
            x = x.to(device)
            feat = model(x).squeeze()  # (B, 512, 1, 1) → (B, 512)
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            features.append(feat.cpu().numpy())
            filenames.extend(names)
    return np.concatenate(features), filenames

# ------------------------
# 쿼리 이미지 feature 추출
# ------------------------
X_query, filenames = extract_features(query_loader)

# ------------------------
# 기존 학습 feature 로드 (X_train, y_train)
# → 이미 추출해둔 경우 아래처럼 불러오면 됨
np.load('X_train.npy'), np.load('y_train.npy')
# ------------------------
# 예시로 더미 생성 (실제에선 바꿔야 함)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# 아래는 테스트용 더미 (삭제하고 위에 주석 해제할 것)
# 10개 클래스 * 100개씩
X_train = np.random.rand(1000, 512)
y_train = np.repeat(np.arange(10), 100)

# ------------------------
# KNN Top-10 예측
# ------------------------
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(X_train)

distances, indices = knn.kneighbors(X_query)  # (N_query, 10)
top10_preds = indices

# ------------------------
# 라벨 이름 변환
# ------------------------
top10_label_names = [[class_names[y_train[j]] for j in row] for row in top10_preds]

# ------------------------
# 결과 저장
# ------------------------
with open("c2_t2_a1.csv", "w", newline='') as f:
    for filename, labels in zip(filenames, top10_label_names):
        f.write(",".join([filename] + labels) + "\n")

print("✅ Saved: c2_t2_a1.csv (Challenge 2 - Task 2)")

