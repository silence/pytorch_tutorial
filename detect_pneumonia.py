#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

data_dir = '/Users/apple/Downloads/dataset/chest_xray/'
TEST = 'test'
TRAIN = 'train'
# VAL = 'val'
BATCH_SIZE = 10
torch.manual_seed(1)
EPOCH = 10
LR = 0.001

data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),              # ( H x W x C) transform to (C x H x W) and normalize to [0,1]
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir,x),transform = data_transforms[x])
    for x in [TRAIN,TEST]
}

data_loaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],batch_size = BATCH_SIZE,
        shuffle = True, num_workers = 4
    )
    for x in [TRAIN,TEST]
}

data_sizes = {x: len(image_datasets[x]) for x in [TRAIN,TEST]}

for x in [TRAIN,TEST]:
    print('Loaded {} images under {} folder'.format(data_sizes[x],x))

print('Classes: ')
class_names = image_datasets[TRAIN].classes
print(class_names)

def imshow(inp,title=None):
    inp = inp.numpy().transpose((1,2,0))  # restore tensor image (C x H x W) to ndarray image (H x W x C)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title,fontdict={'fontsize':5})
    plt.pause(0.001)

def show_databatch(inputs,classes):
    out = torchvision.utils.make_grid(inputs,nrow=BATCH_SIZE)
    imshow(out,title=[class_names[x] for x in classes])

inputs, classes = next(iter(data_loaders[TRAIN]))
show_databatch(inputs,classes)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential( # input shape (3,224,224)
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride = 1,         # filter step
                padding = 2,
            ), # output shape (16,224,224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output shape (16,112,112)
        )
        self.conv2 = nn.Sequential(      #input shape (16,112,112)
            nn.Conv2d(16,32,5,1,2),       #output shape (32,112,112)
            nn.ReLU(),
            nn.MaxPool2d(2),              #output shape (32,56,56)
        )
        self.out1 = nn.Linear(32*56*56,1000)
        self.out2 = nn.Linear(1000,16)
        self.out3 = nn.Linear(16,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)  # output shape (batch_size, 32*56*56)
        output = self.out1(x)
        output = self.out2(output)
        output = self.out3(output)
        return output

cnn = CNN()
print(cnn)

optimizer = torch.optim.SGD(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(data_loaders[TRAIN]):
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            accuracy = 0
            for test_x , test_y in iter(data_loaders[TEST]):
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy += float(torch.sum(pred_y == test_y)) / float(data_sizes[TEST])

            print('Epoch: {} | train loss: {:.4f} | test accuracy: {:.2f}'.format(epoch, loss.data.numpy(), accuracy))
