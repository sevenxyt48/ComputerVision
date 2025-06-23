import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm
import os

# ------------------------
# 환경 설정
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "./recaptcha-dataset/Large"  # 학습용 데이터 디렉토리

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
# 학습용 데이터 로딩
# ------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
class_names = train_dataset.classes  # ['Bicycle', 'Bridge', ..., 'Traffic Light']

# ------------------------
# 사전학습 ResNet18 모델
# ------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])  # FC 제거
model.to(device)
model.eval()

# ------------------------
# Feature 추출 함수
# ------------------------
def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            feat = model(x).squeeze()  # (B, 512, 1, 1) → (B, 512)
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            features.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# ------------------------
# 학습 데이터에서 feature 추출
# ------------------------
X_train, y_train = extract_features(train_loader)

# ------------------------
# 저장
# ------------------------
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

print("✅ Saved: X_train.npy, y_train.npy")
