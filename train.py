import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pth', type=str, default=None, help='Path to the model checkpoint file')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--test_batches', type=int, default=100, help='Number of batches when testing')
parser.add_argument('--drop_out_p', type=float, default=0.2, help='Dropout probability')
args = parser.parse_args()

if os.path.exists("pth") == False:
    os.mkdir("pth")

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载并预处理CIFAR-100数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 期望的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=True)

# 3. 定义ViT模型
best_prec, epoch = 0, 0

model = swin_v2_t()

if args.pth is not None: 
    model.head = nn.Linear(model.head.in_features, 100)
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            # print(f"Before: {module}")
            module.p = args.drop_out_p  # 修改 dropout 概率
            # print(f"After: {module}")
    load = torch.load(args.pth)
    print("load model from: ", args.pth)
    print(f"epoch: {load['epoch']}, dataset: {load['dataset']}, prec: {load['prec']}, isbest: {load['isbest']}")
    model.load_state_dict(load['model'])
    best_prec = load['prec']
    epoch = load['epoch']
else:
    weights = Swin_V2_T_Weights.DEFAULT
    model.load_state_dict(weights.get_state_dict())
    model.head = nn.Linear(model.head.in_features, 100)  # 修改分类头
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            # print(f"Before: {module}")
            module.p = args.drop_out_p  # 修改 dropout 概率
            # print(f"After: {module}")

# 如果有可用的GPU，则将模型转到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def save_checkpoint(state, isbest, filename):
    if isbest:
        path = os.path.join("pth", filename + "_best.pth")
        torch.save(state, path)
    
    path = os.path.join("pth", filename + ".pth")
    torch.save(state, path)

# 6. 评估模型
def evaluate(model, testloader, device, num):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            cur = (predicted == labels).sum().item()
            correct += cur
            if i % 10 == 0:
                print(f"{i}: {cur/labels.size(0)}")
            if i == num:
                break
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    return 100 * correct // total
    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# 5. 训练模型
for t in range(args.epochs):  # 遍历数据集多次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        accu = (torch.max(outputs.data, 1)[1] == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 每10个批次打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f} accu: {accu/64}')
            running_loss = 0.0
         
    print("start evaluating")
    prec = evaluate(model, testloader, device, args.test_batches)
    isbest = False
    if prec > best_prec:
        best_prec = prec
        isbest = True
    save_checkpoint(
        state={
            'epoch': epoch,
            'dataset': 'CIFAR100',
            'prec': prec,
            'isbest': isbest,
            'model': model.state_dict(),
        },
        isbest=isbest,
        filename="swin-transformer"
    )
    epoch += 1

print('Finished Training')
