import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


# 이미지 전처리를 위한 transform 설정
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.img_names = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]
        self.class_to_idx = {'apple': 0, 'banana': 1, 'orange': 2} # 예시, 실제 클래스에 맞게 수정 필요
        self.classes = list(self.class_to_idx.keys())


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        annotation_path = os.path.join(self.annotations_dir, self.img_names[idx].replace('.jpg', '.xml'))
        image = Image.open(img_path).convert('RGB')
        label = self.extract_label(annotation_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def extract_label(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        label = root.find('object').find('name').text
        return self.class_to_idx[label]

# 데이터셋 및 데이터 로더 인스턴스화
train_dataset = CustomDataset('../데이터셋/train_zip/train/', '../데이터셋/train_zip/train/', transform=transform)
test_dataset = CustomDataset('../데이터셋/test_zip/test/', '../데이터셋/test_zip/test/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class VGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()
        # VGG-16의 합성곱 계층과 맥스풀링 계층을 구성하는 부분
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 두 번째 블록
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 세 번째 블록
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size= 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 네 번째 블록
            nn.Conv2d(256, 512, kernel_size = 3 , padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size= 3 , padding= 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size= 3 , padding= 1 ),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size= 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size= 2, stride = 2),
            # 다섯 번째 블록
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size= 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
        )
        # VGG-16의 완전 연결 계층 부분
        self.classifier = nn.Sequential(
            # 입력이미지 224 maxpooling 5번 이므로, 224/2**5 = 7 = 512 * 7 * 7 = 25088
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 중간 레이어의 출력을 추출하고자 할 때
        x = self.features[0:5](x)  # 첫 번째 블록
        features_conv2 = x
        x = self.features[5:10](x)  # 두 번째 블록
        x = self.features[10:18](x)  # 세 번째 블록
        features_conv3 = x
        x = self.features[18:](x)  # 나머지 블록
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, features_conv2, features_conv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def visualize_feature_map(feature_map_batch, img_names, num_images=1, num_columns=8):
    for img_index in range(num_images):
        # 배치에서 선택된 이미지의 특징 맵을 가져옵니다.
        feature_map = feature_map_batch[img_index]
        num_feature_maps = feature_map.size(0)
        num_rows = int(np.ceil(num_feature_maps / num_columns))
        
        # 선택된 이미지에 대한 특징 맵 시각화
        plt.figure(figsize=(num_columns * 2, num_rows * 2))
        plt.suptitle(f"Feature Maps for Image: {img_names[img_index]}", fontsize=16)
        for i in range(num_feature_maps):
            plt.subplot(num_rows, num_columns, i+1)
            plt.imshow(feature_map[i].cpu().detach().numpy(), cmap='viridis')
            plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# def visualize_layer_output(model, layer_num, data_loader):
#     images, _ = next(iter(data_loader))  # 데이터 로더에서 첫 번째 배치를 가져옴

#     images = images.to(device)  # 입력 이미지를 GPU로 이동시키는 부분을 추가

#     with torch.no_grad():  # 그라디언트 계산을 중지하여 메모리 사용량을 줄임
#         x = images
#         for i, layer in enumerate(model.features):
#             x = layer(x)
#             if i == layer_num:
#                 break
        
#         # 선택된 레이어의 출력 시각화
#         out_features = x
#         fig, axes = plt.subplots(1, min(4, out_features.shape[1]), figsize=(20, 5))
#         for i, ax in enumerate(axes):
#             ax.imshow(out_features[0, i].cpu().numpy(), cmap='viridis')  # GPU 텐서를 CPU로 이동시키고 numpy 배열로 변환
#             ax.axis('off')
#         plt.show()


# 훈련 과정에서 중간 레이어의 특징 맵을 시각화합니다.
def train_model(model, criterion, optimizer, num_epochs=10, visualize=False):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, features_conv2, features_conv3 = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            if visualize and i == 0:  # 첫 번째 배치에 대해 시각화
                img_names = [train_dataset.img_names[j] for j in range(inputs.size(0))]
                visualize_feature_map(features_conv2, img_names)
                # visualize_layer_output(model, 10, train_loader)
                visualize_feature_map(features_conv3, img_names)
                # visualize_layer_output(model, 17, train_loader)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 모델, 손실 함수, 최적화 알고리즘 초기화
model = VGG16(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련 및 첫 번째 배치의 중간 레이어 시각화
train_model(model, criterion, optimizer, num_epochs=10, visualize=True)


# 최종 test 이미지 측정 후 결과 출력
def test_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():  # 검증을 위해 그라디언트를 종료하고, 메모리 사용량을 줄임
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # 결과 확인
    for i in range(5):
        print(f'Image: {test_dataset.img_names[i]} - Real: {test_dataset.classes[all_labels[i]]}, Predicted: {test_dataset.classes[all_preds[i]]}')
    return test_loss, test_acc

# Mooel load
test_loss, test_acc = test_model(model, test_loader, criterion)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# 테스트 데이터셋에서 이미지 및 레이블 가져오기
images, labels = next(iter(test_loader))

# 모델을 사용하여 예측
outputs, _, _ = model(images.to(device))
_, preds = torch.max(outputs, 1)

# 이미지 및 예측 결과 시각화
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(5):  # 5개의 이미지 샘플을 시각화
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(images.cpu().data[idx])
    ax.set_title(f"Real: {test_dataset.classes[labels[idx]]}\nPredicted: {test_dataset.classes[preds[idx].cpu().numpy()]}")
plt.show()