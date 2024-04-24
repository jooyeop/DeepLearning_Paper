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
import torch.nn.functional as F

# 이미지 리사이징 및 정규화를 위한 트랜스폼
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 입력 크기 조정
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
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

# Inception_ResNet-v2 STEM
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        # 첫 번째 블록
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0)  # padding을 제거 (필요하지 않음)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) # padding을 제거 (필요하지 않음)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=0) # padding을 제거 (필요하지 않음)

        # 두 번째 블록의 우측 가지
        self.conv5 = nn.Conv2d(160, 64, kernel_size=1, stride=1, padding=0) # padding 조정
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.conv8 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0)  # stride를 1로 조정

        # 두 번째 블록의 좌측 가지
        self.conv9 = nn.Conv2d(160, 64, kernel_size=1, stride=1, padding=0) # padding 조정
        self.conv10 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0) # padding 추가

        # 세 번째 블록
        self.conv11 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0) # padding 추가
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 0)      # padding 추가
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding = 0)     # padding 추가

    def forward(self, x):
        # 첫 번째 블록
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 두 번째 블록의 최대 풀링과 합치기
        x0 = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = torch.cat((x, x0), 1) # 최대 풀링 결과와 첫 번째 블록 결과 합치기
        
        # 두 번째 블록의 우측 가지
        x1 = F.relu(self.conv5(x))
        x1 = F.relu(self.conv6(x1))
        x1 = F.relu(self.conv7(x1))
        x1 = F.relu(self.conv8(x1))
        
        # 두 번째 블록의 좌측 가지
        x2 = F.relu(self.conv9(x))
        x2 = F.relu(self.conv10(x2))
        
        # 두 가지 결과 합치기
        x = torch.cat((x1, x2), 1)
        
        # 세 번째 블록
        x1 = F.relu(self.conv11(x))
        x2 = self.maxpool2(x)
        x = torch.cat((x1, x2), 1)  # 세 번째 블록 결과 합치기
        # print(x.shape)
        
        return x
    
# 5x Inception-ResNet-A 블록
class Inception_ResNet_A(nn.Module):
    def __init__(self):
        super(Inception_ResNet_A, self).__init__()
        # 좌측 가지
        self.conv1 = nn.Conv2d(384, 32, kernel_size=1, stride=1, padding=0)
        # 중앙 가지
        self.conv2 = nn.Conv2d(384, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # 우측 가지
        self.conv4 = nn.Conv2d(384, 32, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)

        # 1x1 Conv 384 Linear
        self.conv7 = nn.Conv2d(128, 384, kernel_size=1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 좌측 가지
        x1 = F.relu(self.conv1(x))
        
        # 중앙 가지
        x2 = F.relu(self.conv2(x))
        x2 = F.relu(self.conv3(x2))
        
        # 우측 가지
        x3 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x3))
        x3 = F.relu(self.conv6(x3))

        # Linear
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv7(x)
        x = self.bn(x)
        x = self.relu(x + self.shortcut(x))
        # print(x.shape)
        
        return x

# Reduction-A 블록
class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        # 좌측 가지 (3x3 Max Pooling), stride2, VALID padding
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        # 중앙 가지 (3x3 컨볼루션), n stride 2, VALID padding
        self.conv1 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=0)
        # 우측 가지 (1x1 Conv) k, (3x3 Conv) l, (3x3 Conv) m, stride 2, VALID padding
        self.conv2 = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # 좌측 가지
        x1 = self.maxpool(x)
        
        # 중앙 가지
        x2 = F.relu(self.conv1(x))
        
        # 우측 가지
        x3 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv4(x3))
        
        # 세 결과 합치기
        x = torch.cat((x1, x2, x3), 1)
        # print(x.shape)
        
        return x
    

# 10x Inception-ResNet-B 블록
class Inception_ResNet_B(nn.Module):
    def __init__(self):
        super(Inception_ResNet_B, self).__init__()
        # 좌측 가지
        self.conv1 = nn.Conv2d(1152, 192, kernel_size=1, stride=1, padding = 0) # 1x1 컨볼루션
        # 우측 가지
        self.conv2 = nn.Conv2d(1152, 128, kernel_size=1, stride=1, padding = 0) # 첫 번째 1x1 컨볼루션
        self.conv3 = nn.Conv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)) # 1x7 컨볼루션
        self.conv4 = nn.Conv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)) # 7x1 컨볼루션

        # 1x1 컨볼루션으로 차원 조정
        self.conv5 = nn.Conv2d(384, 1152, kernel_size=1, stride=1, padding = 0)  # 출력 채널을 원래대로 복원
        self.shortcut = nn.Conv2d(1152, 1152, kernel_size=1, stride=1, padding = 0)
        self.bn = nn.BatchNorm2d(1152)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        x = torch.cat((x1, x2), 1)  # 여기를 수정했습니다.
        x = self.conv5(x) * 0.1
        x = self.bn(x + x_shortcut)
        x = self.relu(x)
        # print(x.shape)
        
        return x

    
# Reduction-B 블록
class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        # 맨좌측 가지 (3x3 Max Pooling), stride2, VALID padding
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        # 좌측가지
        self.conv1 = nn.Conv2d(1152, 256, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(256, 384, kernel_size = 3, stride = 2, padding = 0)
        # 우측 가지
        self.conv3 = nn.Conv2d(1152, 256, kernel_size = 1, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(256, 288, kernel_size = 3, stride = 2, padding = 0)
        # 맨우측 가지
        self.conv5 = nn.Conv2d(1152, 256, kernel_size = 1, stride = 1, padding = 0)
        self.conv6 = nn.Conv2d(256, 288, kernel_size = 3, stride = 1, padding = 1)
        self.conv7 = nn.Conv2d(288, 320, kernel_size = 3, stride = 2, padding = 0)


    def forward(self, x):
        # 맨좌측 가지
        x1 = self.maxpool(x)

        # 좌측 가지
        x2 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x2))

        # 우측 가지
        x3 = F.relu(self.conv3(x))
        x3 = F.relu(self.conv4(x3))

        # 맨우측 가지
        x4 = F.relu(self.conv5(x))
        x4 = F.relu(self.conv6(x4))
        x4 = F.relu(self.conv7(x4))

        # 네 결과 합치기
        x = torch.cat((x1, x2, x3, x4), 1)
        # print(x.shape)
        return x
    
# 5x Inception-ResNet-C 블록
class Inception_ResNet_C(nn.Module):
    def __init__(self):
        super(Inception_ResNet_C, self).__init__()
        # 좌측 가지
        self.conv1 = nn.Conv2d(2144, 192, kernel_size=1, stride=1, padding=0)
        # 우측
        self.conv2 = nn.Conv2d(2144, 192, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv4 = nn.Conv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # 1x1 컨볼루션으로 차원 조정
        self.conv5 = nn.Conv2d(448, 2144, kernel_size=1, stride=1, padding = 0)  # 출력 채널을 원래대로 복원
        self.shortcut = nn.Conv2d(2144, 2144, kernel_size=1, stride=1, padding = 0)
        self.bn = nn.BatchNorm2d(2144)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        x = torch.cat((x1, x2), 1)
        x = self.conv5(x) * 0.1
        x = self.bn(x + x_shortcut)
        x = self.relu(x)
        # print(x.shape)

        return x
    
# Averge Pooling, Dropout, Fully Connected Layer
class Inception_ResNet_v2(nn.Module):
    def __init__(self,A, B, C, num_classes, init_weights=True):
        super().__init__()
        blocks = []
        blocks.append(Stem())
        for i in range(A):
            blocks.append(Inception_ResNet_A())
        blocks.append(Reduction_A())
        for i in range(B):
            blocks.append(Inception_ResNet_B())
        blocks.append(Reduction_B())
        for i in range(C):
            blocks.append(Inception_ResNet_C())
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2144, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x= self.blocks(x)
        view_x = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        logit = x
        x = F.softmax(x, dim=1)
        probas = x
        return x, view_x, logit, probas
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



# 모델 초기화 및 GPU 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
model = Inception_ResNet_v2(10, 20, 10, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

data_loader = train_loader

def visualize_feature_maps(batch_data, titles):
    fig, axarr = plt.subplots(nrows=1, ncols=len(batch_data), figsize=(15, 5))
    for idx, data in enumerate(batch_data):
        ax = axarr[idx] if len(batch_data) > 1 else axarr
        ax.imshow(data, cmap='viridis')
        ax.set_title(titles[idx])
        ax.axis('off')
    plt.show()

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, _, view_x, _ = model(features)  # logits와 probas 변수명을 정정합니다.
        _, predicted_labels = torch.max(logits, 1)  # probas 대신 logits에서 최대 값을 얻습니다.
        
        # 예측이 맞은 횟수를 카운트합니다.
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        if i == 0 and epoch == 0:
            assert view_x.size(1) >= 3, f"view_x has less than 3 channels, it has {view_x.size(1)} channels."
            num_channels_to_visualize = min(view_x.size(1), 4)  # Visualize up to 4 channels.
            titles = ['Channel ' + str(i) for i in range(num_channels_to_visualize)]
            
            feature_maps_to_visualize = []
            for i in range(num_channels_to_visualize):
                feature_map = view_x[0, i].cpu().numpy()  # Convert to numpy array once here.
                
                # Check the shape of feature_map and proceed accordingly.
                if feature_map.ndim == 2:
                    # If the feature_map is 2D (H, W), it's ready for visualization.
                    feature_maps_to_visualize.append(feature_map)
                elif feature_map.ndim == 1:
                    # If the feature_map is 1D (C), reshape it to (1, 1, C).
                    feature_maps_to_visualize.append(feature_map.reshape(1, 1, -1))
                elif feature_map.ndim == 0:
                    # If the feature_map is a scalar, reshape it to (1, 1).
                    feature_maps_to_visualize.append(np.array([[feature_map]]))
                    
            visualize_feature_maps(feature_maps_to_visualize, titles)

    return correct_pred.float()/num_examples * 100

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward and Backprop
        logits, probas, _, _ = model(features)
        cost = criterion(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        # Update model parameters
        optimizer.step()

        # Logging
        if not batch_idx % 50:
            if epoch % 10 == 0 or epoch == NUM_EPOCHS-1:
                print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost))
            
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        if epoch % 10 == 0 or epoch == NUM_EPOCHS-1:
            train_accuracy = compute_accuracy(model, train_loader, device=DEVICE)
            test_accuracy = compute_accuracy(model, test_loader, device=DEVICE)
            print('Epoch: %03d/%03d | Train: %.3f%% | Test: %.3f%%' % (epoch+1, NUM_EPOCHS, train_accuracy, test_accuracy))