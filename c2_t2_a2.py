import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import csv
from tqdm import tqdm

# -------------------------------
# 1. 디바이스 설정
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. 사전학습 모델 불러오기 (FC 제거)
# -------------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

# -------------------------------
# 3. 전처리 파이프라인
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4. 학습 데이터 로딩 및 특징 추출
# -------------------------------
train_data = ImageFolder("./recaptcha-dataset/Large", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
class_names = train_data.classes

def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting train features"):
            images = images.to(device)
            feats = model(images).squeeze()
            if len(feats.shape) == 1:
                feats = feats.unsqueeze(0)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

X_train, y_train = extract_features(train_loader)

# -------------------------------
# 5. KNN 학습
# -------------------------------
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(X_train)

# -------------------------------
# 6. Query 이미지에 대해 Top-10 유사도 검색
# -------------------------------
query_dir = "./query"
query_files = sorted([f for f in os.listdir(query_dir) if f.endswith(".png")])

results = []

for filename in tqdm(query_files, desc="Processing queries"):
    img_path = os.path.join(query_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor).squeeze().cpu().numpy().reshape(1, -1)

    dists, inds = knn.kneighbors(feat)
    top10_labels = [class_names[y_train[i]] for i in inds[0]]
    results.append([filename] + top10_labels)

# -------------------------------
# 7. 결과 저장
# -------------------------------
with open("c2_t2_a2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print("✅ [Saved] c2_t2_a2.csv")
