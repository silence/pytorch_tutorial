import torch
import numpy as np
np_data = np.arange(6).reshape(2,3)
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)

import torch.nn.functional as F

x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
'''
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x_np,y_softplus)
plt.show()
'''

import matplotlib.pyplot as plt
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature = 1,n_hidden = 10,n_output =1)

def save():

    net1 = torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU(),torch.nn.Linear(10,1))

    optimizer = torch.optim.SGD(net1.parameters(),lr=0.2)
    loss_func = torch.nn.MSELoss()

    '''
    plt.ion()
    plt.show()
    '''

    for t in range(1000):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1, 'net.pkl')
    torch.save(net1.state_dict(), 'net_params.pkl')
    plt.subplot(131)
    plt.plot(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)

    '''
    if t%5 ==0:
        plt.cla()
        plt.plot(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=2)
        plt.text(0.5,0,'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
    '''

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    plt.subplot(132)
    plt.plot(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)

def restore_params():
    net3 = torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU(),torch.nn.Linear(10,1))
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    plt.subplot(133)
    plt.plot(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)

save()
restore_net()
restore_params()
plt.show()
