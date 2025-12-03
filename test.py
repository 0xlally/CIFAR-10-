import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 网络结构定义 (必须与训练代码完全一致)
# ==========================================
class BetterNet(nn.Module):
    def __init__(self):
        super(BetterNet, self).__init__()
        
        # 第一层块: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二层块: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第三层块: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层 (输入是 128 * 4 * 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        # 注意：虽然这里写了 dropout，但在 net.eval() 模式下它会自动失效
        x = self.dropout(x) 
        x = self.fc2(x)
        return x

def main():
    # ==========================================
    # 2. 准备工作
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'当前测试运行设备: {device}')

    # 测试集的预处理（不需要翻转和裁剪，只要归一化）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ==========================================
    # 3. 加载模型
    # ==========================================
    # 注意文件名要和你训练保存的名字一致
    PATH = './cifar_betternet.pth' 
    
    print(f"正在加载模型: {PATH} ...")
    
    net = BetterNet()
    
    try:
        net.load_state_dict(torch.load(PATH, map_location=device))
    except FileNotFoundError:
        print(f"错误：找不到文件 {PATH}。请先运行训练脚本生成模型。")
        return
    except RuntimeError as e:
        print(f"加载失败，可能是网络结构定义不匹配。\n详细错误: {e}")
        return

    net.to(device)
    
    # 开启评估模式！这会冻结 BN 层和关闭 Dropout，至关重要
    net.eval() 

    # ==========================================
    # 4. 评估总体准确率
    # ==========================================
    print("开始评估...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'>> 总体准确率: {100 * correct / total:.2f} %')

    # ==========================================
    # 5. 评估每一类的准确率
    # ==========================================
    print("\n正在分析每一类的表现...")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f'类别 {classes[i]:>5s} : {acc:.2f} %')

if __name__ == '__main__':
    main()