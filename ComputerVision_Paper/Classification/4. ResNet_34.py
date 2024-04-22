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

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) :
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale) :
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        # BatchNorm2d를 통해 gradient vanishing 문제를 해결하고, ReLU를 통해 비선형성을 추가한다.
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        # 각 층을 거치면서 224 * 224 -> conv1 = 112 * 112 -> maxpool = 56 * 56 / layer1 = 56 * 56 -> layer2 = 28 * 28 -> layer3 = 14 * 14 -> layer4 = 7 * 7
        # 즉, 최종적으로 7 * 7 크기의 feature map이 생성되기 때문에 avgpooling을 통해 1 * 1로 만들어준 후 fully connected layer를 통과시킨다.
        # self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        #512 * block.expansion은 마지막 BasicBlock의 output channel 수를 의미한다.
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 가중치 초기화 : He initialization, 그라디언트 소실 및 폭주 문제 해결
        # 가중치 초기화 관련 논문 : https://arxiv.org/abs/1502.01852
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1) :
        downsample = None
        # 만약 stride가 1이 아니거나, input과 output의 channel 수가 다른 경우 downsample을 수행한다.
        # input과 output의 channel수가 다른 layer의 첫 블록은 stride를 2로 설정하고, downsample을 수행하며, 그 외의 경우는 downsample을 수행하지 않기 때문에 stride는 1로 설정한다.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 레이어의 첫 번째 블록에서 특징 맵의 크기를 다운샘플링 진행
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion # 다음 레이어의 input channel 수를 위 레이어의 output channel 수로 설정
        for i in range(1, blocks):
            # dimensions match가 맞지 않은 경우여서 zero-padding이 적용됨
            # BasicBlock을 사용하는 경우, 3 x 3 컨볼루션에 padding이 1이 적용이 되기 때문에 dimensions match가 맞지 않는 경우에는 zero-padding을 위해 downsampling에 적용된 Conv2d를 사용한다.
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        # 층마다 모두 시각화를 위해 feature map을 저장한다.
        conv = x
        x = self.bn1(x)
        bn = x
        x = self.relu(x)
        relu = x
        x = self.maxpool(x)
        maxpool = x

        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        layer2 = x
        x = self.layer3(x)
        layer3 = x
        x = self.layer4(x)
        layer4 = x
        # 만약 MNIST처럼 1x1로 되어있다면 avgpooing은 건너뛰어도 됌
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1) # 각 클래스에 대한 확률값을 구하기 위해 1차원으로 펼친다. (flatten)
        logits = self.fc(x) # fully connected layer를 통과시켜 각 클래스에 대한 확률값을 구한다.
        probas = F.softmax(logits, dim=1) # 확률값을 구하기 위해 softmax 함수를 적용한다.
        return logits, probas, conv, bn, relu, maxpool, layer1, layer2, layer3, layer4
    
def resnet34(num_classes, grayscale):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
    return model

# 하이퍼파라미터 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
GRAYSCALE = False
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
model = resnet34(NUM_CLASSES, GRAYSCALE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

data_loader = train_loader

def visualize_feature_maps(batch_data, titles):
    fig, axarr = plt.subplots(nrows=1, ncols=len(batch_data), figsize=(15, 5))
    for idx, data in enumerate(batch_data):
        ax = axarr[idx] if len(batch_data) > 1 else axarr
        ax.imshow(data[0, 0].cpu().numpy())  # Assuming data is [C, H, W] and we visualize the first channel
        ax.set_title(titles[idx])
        ax.axis('off')
    plt.show()

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)
        logits, probas, conv, bn, relu, maxpool, layer1, layer2, layer3, layer4 = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        if i == 0 and epoch == 0 :
            visualize_feature_maps([conv, bn, relu, maxpool, layer1, layer2, layer3, layer4],
                                   ['conv', 'bn', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'])
    return 100-(correct_pred.float()/num_examples * 100)
    


for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas, conv, bn, relu, maxpool, layer1, layer2, layer3, layer4 = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            if epoch % 10 == 0 or epoch == NUM_EPOCHS-1 :
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        if epoch % 10 == 0 or epoch == NUM_EPOCHS-1 :
            print('Epoch: %03d/%03d | Train_Error: %.3f%%' % (
                epoch+1, NUM_EPOCHS, 
                compute_accuracy(model, train_loader, device=DEVICE)))