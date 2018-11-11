import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

torch.manual_seed(1)
EPOCH = 1
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train = True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train = False
)

train_loader = Data.DataLoader(dataset= train_data,batch_size=BATCH_SIZE,shuffle = True,num_workers=4)

test_x = torch.unsqueeze(test_data.test_data,dim = 1).type(torch.FloatTensor)[:2000]/255 # shape from (2000,28,28) to (2000,1,28,28) andn normalized to 0:1

test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential( # input shape (1,28,28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride = 1,         # filter step
                padding = 2,
            ), # output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output shape (16,14,14)
        )
        self.conv2 = nn.Sequential(      #input shape (16,14,14)
            nn.Conv2d(16,32,5,1,2),       #output shape (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),              #output shape (32,7,7)
        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)  # output shape (batch_size, 32*7*7)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x,batch_y) in enumerate(train_loader):

        output = cnn(batch_x)
        loss = loss_func(output,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 ==0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = float(torch.sum(pred_y == test_y)) / float(test_y.size(0))
            print('Epoch: {} | train loss: {:.4f} | test accuracy: {:.2f}'.format(epoch,loss.data.numpy(),accuracy))

# print 100 predictions from test data
test_output = cnn(test_x[500:600])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[500:600].numpy(), 'real number')

