# challenge2_task1.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_dir = "./recaptcha-dataset/Large"  # train 데이터는 폴더 구조 있음
val_data_dir = "./query"  # val 데이터는 단일 폴더, 라벨 없음
class_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney',
               'Crosswalk', 'Hydrant', 'Motorcycle', 'Palm', 'Traffic Light']
input_size = 224
batch_size = 32

# 전처리 정의 (train과 동일하게)
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation용 단일 폴더 데이터셋 정의 (라벨 없음)
class QueryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.fnames = sorted([f for f in os.listdir(root_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.fnames[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 라벨 없으므로 0 임시 반환
        return img, 0

# Train 데이터셋 로드
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
n_total = len(train_dataset)
indices = np.random.permutation(n_total)
split = int(n_total * 0.8)
train_idx, val_idx = indices[:split], indices[split:]
train_set = Subset(train_dataset, train_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

# Validation 데이터셋 (query 폴더, 라벨 없음)
val_set = QueryDataset(val_data_dir, transform=transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 사전학습 후 fine-tune된 모델 불러오기
model_ft = models.resnet18(weights=None)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 10)
model_ft.load_state_dict(torch.load("resnet18_ft.pt", map_location=device))

# FC 제거하여 feature extractor 구성
resnet18_feat = nn.Sequential(*list(model_ft.children())[:-1]).to(device)
resnet18_feat.eval()

# Feature 추출 함수
def extract_features(dataloader, with_labels=True):
    features = []
    labels = [] if with_labels else None
    with torch.no_grad():
        for inputs, lbls in tqdm(dataloader):
            inputs = inputs.to(device)
            out = resnet18_feat(inputs).view(inputs.size(0), -1).cpu().numpy()
            features.append(out)
            if with_labels:
                labels.append(lbls.numpy())
    if with_labels:
        return np.concatenate(features), np.concatenate(labels)
    else:
        return np.concatenate(features)

# Train feature, label 추출
X_train, y_train = extract_features(train_loader, with_labels=True)

# Validation feature만 추출 (라벨 없음)
X_val = extract_features(val_loader, with_labels=False)

# KNN 분류기 학습 및 예측
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(X_train, y_train)

preds = knn.predict(X_val)

# 파일명 확장자 제거
val_filenames = [os.path.splitext(f)[0] for f in val_set.fnames]

# 결과 CSV 저장
df = pd.DataFrame({
    "filename": val_filenames,
    "pred_label": [class_names[p] for p in preds]
})
df.to_csv("c2_t1_a1.csv", index=False)

print(f"[Challenge 2 - Task 1] Saved: c2_t1_a1.csv")
