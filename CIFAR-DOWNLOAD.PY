import torchvision
import torchvision.transforms as transforms

# 定义数据预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 下载并加载测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)