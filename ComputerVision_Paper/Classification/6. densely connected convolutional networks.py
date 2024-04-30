import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
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

model_parameters = {}
model_parameters['densenet121'] = [6, 12, 24, 16]
model_parameters['densenet169'] = [6, 12, 32, 32]
model_parameters['densenet201'] = [6, 12, 48, 32]
model_parameters['densenet264'] = [6, 12, 64, 48]

# Growth rate
k = 12
compression_factor = 0.5

class DenseLayer(nn.Module):

    def __init__(self,in_channels):
        super(DenseLayer,self).__init__()

        self.BN1 = nn.BatchNorm2d(num_features = in_channels) # 64, 48, 224, 224
        self.conv1 = nn.Conv2d( in_channels=in_channels , out_channels=4*k , kernel_size=1 , stride=1 , padding=0 , bias = False ) # 64, 48, 224, 224

        self.BN2 = nn.BatchNorm2d(num_features = 4*k) # 64, 48, 224, 224
        self.conv2 = nn.Conv2d( in_channels=4*k , out_channels=k , kernel_size=3 , stride=1 , padding=1 , bias = False ) # 64, 12, 224, 224

        self.relu = nn.ReLU() # 64, 12, 224, 224

    def forward(self,x):

        xin = x

        # BN -> relu -> conv(1x1)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1(x)

        # BN -> relu -> conv(3x3)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = torch.cat([xin,x],1)

        return x

class DenseBlock(nn.Module):
    def __init__(self,layer_num,in_channels):

        super(DenseBlock,self).__init__()
        self.layer_num = layer_num
        self.deep_nn = nn.ModuleList()

        for num in range(self.layer_num):
            self.deep_nn.add_module(f"DenseLayer_{num}",DenseLayer(in_channels+k*num))


    def forward(self,x):
        xin = x

        for layer in self.deep_nn:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self,in_channels,compression_factor):

        super(TransitionLayer,self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = int(in_channels*compression_factor) ,kernel_size = 1 ,stride = 1 ,padding = 0, bias=False )
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self,x):
        x = self.BN(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x
 


class DenseNet(nn.Module):
    def __init__(self,densenet_variant,in_channels,num_classes=3):
        super(DenseNet,self).__init__()

        # 7x7 conv with s=2 and maxpool
        self.conv1 = nn.Conv2d(in_channels=in_channels ,out_channels=64 ,kernel_size=7 ,stride=2 ,padding=3 ,bias = False)
        self.BN1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        # adding 3 DenseBlocks and 3 Transition Layers 
        self.deep_nn = nn.ModuleList()
        dense_block_inchannels = 64

        for num in range(len(densenet_variant))[:-1]:

            self.deep_nn.add_module( f"DenseBlock_{num+1}" , DenseBlock( densenet_variant[num] , dense_block_inchannels ) )
            dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[num])

            self.deep_nn.add_module( f"TransitionLayer_{num+1}" , TransitionLayer( dense_block_inchannels,compression_factor ) )
            dense_block_inchannels = int(dense_block_inchannels*compression_factor)

        # adding the 4th and final DenseBlock
        self.deep_nn.add_module( f"DenseBlock_{num+2}" , DenseBlock( densenet_variant[-1] , dense_block_inchannels ) )
        dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[-1])

        self.BN2 = nn.BatchNorm2d(num_features=dense_block_inchannels)

        # Average Pool
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        
        # fully connected layer
        self.fc1 = nn.Linear(dense_block_inchannels, num_classes)


    def forward(self,x):
        x = self.relu(self.BN1(self.conv1(x)))
        x = self.maxpool(x)
        layers = []
        
        for layer in self.deep_nn:
            # if not os.path.exists('DenseNet_Layer.txt'):
            #     with open('DenseNet_Layer.txt', 'w'): pass
            # with open('DenseNet_Layer.txt', 'a') as f:
            #     f.write(str(layer))
            #     f.write('\n')
            x = layer(x)
            layers.append(x)
            
        x = self.relu(self.BN2(x))
        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        final = F.softmax(x, dim=1)

        return x, final, layers

def DenseNet121(in_channels,num_classes):
    return DenseNet(model_parameters['densenet121'],in_channels,num_classes)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
NUM_CLASSES = 3
NUM_EPOCHS = 30
model = DenseNet121(3,NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

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
        logits, final, layers = model(features)
        _, predicted_labels = torch.max(final, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        if i == 0:
            visualize_feature_maps([layer[0, 0].detach().cpu().numpy() for layer in layers], 
                                   ['DenseBlock 1', 'DenseBlock 2', 'DenseBlock 3', 'DenseBlock 4', 'DenseBlock 5', 'DenseBlock 6', 'DenseBlock 7'])
    return 100-(correct_pred.float()/num_examples * 100)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, final, layers = model(features)
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