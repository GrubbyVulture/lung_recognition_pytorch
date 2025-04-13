# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms, datasets, models, utils
from torchsummary import summary # 可视化训练过程
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image

image_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(size=300, scale=(0.8, 1.1)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

    'val' : transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],# mean
                             [0.229, 0.224, 0.225])# std
    ]),

    'test': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ])
}

#加载数据集
#路径
data_dir='./chest_xray/'
train_dir=data_dir + 'train/'
val_dir=data_dir + 'val/'
test_dir=data_dir + 'test/'

datasets = {
    'train' : datasets.ImageFolder(train_dir,
                                   transform=image_transforms['train']),
    'val' : datasets.ImageFolder(val_dir,
                                 transform=image_transforms['val']),
    'test' : datasets.ImageFolder(test_dir,
                                  transform=image_transforms['test']),
}

# 定义BATCH_SIZE
BATCH_SIZE = 128 # 每批读取128张图片

# DataLoader : 创建iterator, 按批读取数据
dataloaders = {
    'train' : DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True), # 训练集
    'val' : DataLoader(datasets['val'], batch_size=BATCH_SIZE, shuffle=True), # 验证集
    'test' : DataLoader(datasets['test'], batch_size=BATCH_SIZE, shuffle=True) # 测试集
}

LABEL = dict((v, k) for k, v in datasets['train'].class_to_idx.items())
print(LABEL)

# print(dataloaders['train'].dataset)
# print(dataloaders['train'].dataset.classes)
# print(dataloaders['train'].dataset.root)

# 肺部正常图片
files_normal = os.listdir(os.path.join(str(dataloaders['train'].dataset.root), 'NORMAL'))
# print(files_normal)

# 肺部感染的图片
files_pneumonia = os.listdir(os.path.join(str(dataloaders['train'].dataset.root), 'PNEUMONIA'))

# print(dataloaders['val'].dataset)
# print(dataloaders['test'].dataset)

# 导入SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# SummaryWriter() 向事件文件写入事件和概要

# 日志路径
log_path = '/ligdir/'

def tb_writer():
    time_str = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_path+time_str)
    return writer

writer = tb_writer()

# 一种显示部分图片集的方法
images, labels = next(iter(dataloaders['train'])) # 获取到一批数据

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

grid = utils.make_grid(images)
imshow(grid)

# 在summary中添加图片数据
writer.add_image('X-Ray grid', grid, 0)

writer.flush()

# 获取一张图片的tensor
print(dataloaders['train'].dataset[4])



# 显示一张图片的方法
def show_sample(img, label):
    print("labels : ", dataloaders['train'].dataset.classes[label])
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img*std + mean
    img =np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

show_sample(*dataloaders['train'].dataset[4])



# 显示一张图片的方法
def show_image(img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

one_img = Image.open(dataloaders['train'].dataset.root+'NORMAL/IM-0239-0001.jpeg')
show_image(one_img)


# 记录错误分类的图片
def misclassified_images(pred, writer, target, images, output, epoch, count=10):
    misclassified = (pred != target.data)
    for index, image_tensor in enumerate(images[misclassified][:count]):
        img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, LABEL[pred[misclassified].tolist()[index]],
                                                              LABEL[target.data[misclassified].tolist()[index]])
        writer.add_image(img_name, image_tensor, epoch)


# 自定义池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        size = size or (1, 1)
        self.avgPoolong = nn.AdaptiveAvgPool2d(size)
        self.maxPoolong = nn.AdaptiveMaxPool2d(size)
    def forward(self, x):
        return torch.cat([self.avgPoolong(x), self.maxPoolong(x)], dim=1)


# 迁移学习
def get_model():
    model_dir = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory at: {model_dir}")

    # 优先加载 model 目录下的 model.pth
    model_path = os.path.join(model_dir, 'model.pth')
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print(f"Loaded existing model from: {model_path}")
        return model

    # 如果 model.pth 不存在，继续构建模型
    torch.hub.set_dir(model_dir)
    model = models.resnet50(pretrained=True)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换 avgpool 层
    model.avgpool = AdaptiveConcatPool2d()

    # 修改全连接层
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1)
    )

    # 新增分支：检查 model_para 文件夹中的 model.pth
    model_para_dir = os.path.join(os.getcwd(), 'model_para')
    model_para_path = os.path.join(model_para_dir, 'model.pth')
    if os.path.exists(model_para_path):
        # 加载参数到新构建的模型中
        model.load_state_dict(torch.load(model_para_path))
        print(f"Loaded parameters from model_para: {model_para_path}")

    print(f"Model downloaded to: {model_dir}")
    return model

# 定义训练函数
def train_val(model, device, train_loader, val_loader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0
    val_loss = 0
    val_acc = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)

    train_loss = total_loss/len(train_loader.dataset)
    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.flush()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()*images.size(0)
            _, pred = torch.max(outputs, 1)
            correct = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            val_acc += accuracy.item() * images.size(0)

        val_loss = val_loss/len(val_loader.dataset)
        val_acc = val_acc / len(val_loader.dataset)

    return train_loss, val_loss, val_acc


# 定义测试函数
def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            misclassified_images(pred, writer, labels, images, outputs, epoch)

        avg_loss = total_loss/len(test_loader.dataset)
        accuracy = 100*correct / len(test_loader.dataset)
        writer.add_scalar("Test Loss", total_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.flush()
        return total_loss, accuracy


# 定义训练流程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device.type)

model=get_model().to(device)

# 损失函数
criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练流程
def train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer):
    # 输出信息
    print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format('Epoch', 'Train Loss', 'val_loss', 'val_acc', 'Test Loss', 'Test_acc'))

    # 初始化最小损失
    best_loss = np.inf

    # 开始训练
    for epoch in range(epochs):
        train_loss, val_loss, val_acc = train_val(model, device, dataloaders['train'], dataloaders['val'], optimizer, criterion, epoch, writer)
        # 测试，return: loss + accuracy
        test_loss, test_acc = test(model, device, dataloaders['test'], criterion, epoch, writer)
        if test_loss < best_loss:
            best_loss = test_loss # 保存最小损失
            torch.save(model.state_dict(), './model_para/model_py.pth')

        # 输出结果
        print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc))
        writer.flush()

# 调用函数
epochs=10
train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer)
writer.close()


def plot_confusion(cm):
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues) # 参数设置
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=14)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=14)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.show()

def accuracy(outputs, labels):
    # 计算正确率
    _, preds = torch.max(outputs, dim=1)
    correct = torch.tensor(torch.sum(preds == labels).item() / len(preds))
    return correct

def metrics(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    # precision, recall, F1
    # 混淆矩阵
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    # 绘制混淆矩阵
    plot_confusion(cm)
    # 获取tn, fp, fn, tp
    tn, fp, fn, tp = cm.ravel()
    # 精准率
    precision = tp / (tp + fp)
    # 召回率
    recall = tp / (tp + fn)
    # f1 score
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1

# 计算testloader
precisions = []
recalls = []
f1s = []
accuracies = []

with torch.no_grad():
    model.eval()
    for datas, labels in dataloaders['test']:
        datas, labels = datas.to(device), labels.to(device)
        # 预测输出
        outputs = model(datas)
        # 计算metrics
        precision, recall, f1 = metrics(outputs, labels)
        acc = accuracy(outputs, labels)
        # 保存结果
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(acc.item())


print('精确率：' + ', '.join(['{:.2f}%'.format(pre*100) for pre in precisions]))
print('召回率：' + ', '.join(['{:.2f}%'.format(r*100) for r in recalls]))

print('准确率：' + ', '.join(['{:.2f}%'.format(a*100) for a in accuracies]))