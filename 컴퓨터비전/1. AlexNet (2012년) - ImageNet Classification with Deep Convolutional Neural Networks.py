import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 첫 번째 합성곱 계층: 입력 채널 3(이미지의 RGB), 출력 채널 64, 커널 크기 11, 스트라이드 4, 패딩 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True) # 활성화 함수로 ReLU 사용
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2) # 최대 풀링 계층
        # 두 번째 합성곱 계층: 입력 채널 64, 출력 채널 192, 커널 크기 5, 패딩 2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        # 세 번째 합성곱 계층: 입력 채널 192, 출력 채널 384, 커널 크기 3, 패딩 1
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        # 네 번째 합성곱 계층: 입력 채널 384, 출력 채널 256, 커널 크기 3, 패딩 1
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # 다섯 번째 합성곱 계층: 입력 채널 256, 출력 채널 256, 커널 크기 3, 패딩 1
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.adapt_pool = nn.AdaptiveAvgPool2d((6, 6)) # 적응형 평균 풀링 계층
        # 완전 연결 계층
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout = nn.Dropout() # 드롭아웃 계층
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes) # 출력 계층

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1) # 플래튼 계층
       
