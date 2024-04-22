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

class Bottleneck(nn.Module) :
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None) :
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size = 1, bias =False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) :
        residual = x
        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1 conv * 4
        out = self.conv3(out)
        out = self.bn3(out)
        # skip connection
        if self.downsample is not None :
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
        
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
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
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas, conv, bn, relu, maxpool, layer1, layer2, layer3, layer4
    
def resnet50(num_classes, grayscale):
    """Constructs a ResNet-50 model."""
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale)
    return model

# 하이퍼파라미터 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
GRAYSCALE = False
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
model = resnet50(NUM_CLASSES, GRAYSCALE).to(DEVICE)
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