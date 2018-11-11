from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# 交互模式
plt.ion()

data_transforms = {  # 图片转换
    'train': transforms.Compose([  # 训练集
        transforms.RandomResizedCrop(224),  # 将图片随机剪切成224*224大小
        transforms.RandomHorizontalFlip(),  # 对图片随机水平翻转
        transforms.ToTensor(),  # 将图片转化成Tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 对图片的 C,H,W 分别正则化，mean分别为0.485,0.456,0.406, std 分别为0.229, 0.224, 0.225
    ]),
    'val': transforms.Compose([  # 验证集
        transforms.Resize(256),  # 将图片调整为256*256的大小
        transforms.CenterCrop(224),  # 裁剪图片中间的224*224的部分
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = '/Users/apple/Downloads/dataset/hymenoptera_data'  # 数据集目录名，下载自https://download.pytorch.org/tutorial/hymenoptera_data.zip

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'val']}  # 导入训练集和验证集图片地址和transforms
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
               ['train', 'val']}  # 使用DataLoader导入数据
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}  # 数据集大小
class_names = image_datasets['train'].classes  # 训练集类名

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 是否使用gpu


def imshow(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))  # 重新排列图像数组维度，将 C,H,W 转化为 H,W,C
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # 对图像数组normalize还原
    inp = np.clip(inp, 0, 1)  # 设置数组里最大值为1，最小值为0
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 图片更新时暂停一会


inputs, classes = next(iter(dataloaders['train']))  # 获取一批训练数据
out = torchvision.utils.make_grid(inputs)  # 对图片进行网格展示
imshow(out, title=[class_names[x] for x in classes])  # 展示图片和title


def train_model(model, criterion, optimzer, scheduler, num_epochs=25):
    since = time.time()  # 开始时间
    best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝状态字典
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step() # 使用最优化算法更新权重值
                model.train()  # 设置为训练模型
            else:
                model.eval()  # 设置为评价模型

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                # 导入输入和标签
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimzer.zero_grad()  # 设置初始梯度值为0

                with torch.set_grad_enabled(phase == 'train'):
                    # 使用模型计算出输出值、预测值
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # 使用损失函数计算出loss

                    if phase == 'train':
                        loss.backward() # 反向传播
                        optimzer.step() # 更新权重

                running_loss += loss.item() * inputs.size(0) # 以列表形式返回总的loss
                running_corrects += torch.sum(preds == labels.data) # 计算预测正确率

            epoch_loss = running_loss / dataset_sizes[phase] # 每一轮的loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # 每一轮的正确率

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 深度复制最优模型权重值
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time().time() - since  # 训练用时
    print('Train complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # 加载最优模型权重值
    model.load_state_dict(best_model_wts)
    return model

# 可视化训练结果
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval() # 设置为评价模式
    images_so_far = 0 # 到目前为止的图片数
    fig = plt.figure() # 画出空图底

    with torch.no_grad():
        # 迭代验证集
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 使用模型计算输出结果和预测结果
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 对每张图片进行展示
            for j in range(inputs.size()[0]):
                images_so_far += 1
                # 以两列展示图片
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                # 不绘制坐标
                ax.axis('off')
                # 绘制每张图的标题
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                # 展示图片
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(pretrained=True) # 加载预训练的模型
num_ftrs = model_ft.fc.in_features # 模型的特征数
model_ft.fc = nn.Linear(num_ftrs, 2) # 重置全连接层，输入节点数为特征数，输出节点为2

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) # 随机梯度下降最优化算法

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 每7轮衰减学习率为原来的0.1

# 训练模型
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

# 可视化模型
visualize_model(model_ft)

model_conv = torchvision.models.resnet18(pretrained=True)
# 冻结除了全连接层以外的层的权重
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9) # 随机梯度下降最优化算法，学习率为0.001,动量为0.9
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
visualize_model(model_conv)

plt.ioff()
plt.show()
