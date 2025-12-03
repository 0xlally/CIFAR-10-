import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on: {device}')

    # ==========================================
    # 策略 1: 数据增强 (Data Augmentation)
    # ==========================================
    # 训练集：加了随机翻转和随机裁剪，增加难度，防止死记硬背
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 50%概率水平翻转
        transforms.RandomCrop(32, padding=4), # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 测试集：保持原样，只做标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    # Batch Size 建议 64 或 128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # ==========================================
    # 策略 2: 定义更深更宽的网络 (Better Architecture)
    # ==========================================
    class BetterNet(nn.Module):
        def __init__(self):
            super(BetterNet, self).__init__()
            
            # 第一层块: 3 -> 32
            # 加了 BatchNorm2d，这是提升准确率的神器
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            
            # 第二层块: 32 -> 64
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            
            # 第三层块: 64 -> 128
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.pool = nn.MaxPool2d(2, 2)
            
            # 全连接层
            # 经过3次池化(32->16->8->4)，且最后一层通道是128
            # 所以输入是 128 * 4 * 4
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
            
            # Dropout: 防止过拟合，扔掉一些神经元
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            # Conv -> BN -> ReLU -> Pool
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            
            x = torch.flatten(x, 1)
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x) # 在全连接层之间加 Dropout
            x = self.fc2(x)
            return x

    net = BetterNet()
    net.to(device)

    # ==========================================
    # 策略 3: 使用 Adam 优化器
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    # Adam 通常比 SGD 收敛更快，且对学习率不那么敏感
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # ==========================================
    # 训练循环
    # ==========================================
    # 建议跑 20-30 个 Epoch 看看效果
    for epoch in range(50): 
        net.train() # 确保开启训练模式(启用 Dropout/BN)
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 这里的 200 是打印频率，你可以根据自己喜好改
            if i % 200 == 199:
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    # 保存新模型
    torch.save(net.state_dict(), './cifar_betternet.pth')
    print("BetterNet 模型已保存")

if __name__ == '__main__':
    main()