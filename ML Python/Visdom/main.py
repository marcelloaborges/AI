import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets

import visdom
vis = visdom.Visdom()

# vis.text("Hello, world!",env="main")

# a=torch.randn(3,200,200)
# vis.image(a)

# vis.images(torch.Tensor(3,3,28,28))

# MNIST = dsets.MNIST(root="./MNIST_data",train = True,transform=torchvision.transforms.ToTensor(), download=True)
# cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)

# data = cifar10.__getitem__(0)
# print(data[0].shape)
# vis.images(data[0],env="main")

# data = MNIST.__getitem__(0)
# print(data[0].shape)
# vis.images(data[0],env="main")

# data_loader = torch.utils.data.DataLoader(dataset = MNIST, batch_size = 32, shuffle = False)

# for num, value in enumerate(data_loader):
#     value = value[0]
#     print(value.shape)
#     vis.images(value)
#     break

# # vis.close(env="main")

# Y_data = torch.randn(5)
# plt = vis.line (Y=Y_data)

# X_data = torch.Tensor([1,2,3,4,5])
# plt = vis.line(Y=Y_data, X=X_data)

# Y_append = torch.randn(1)
# X_append = torch.Tensor([6])

# vis.line(Y=Y_append, X=X_append, win=plt, update='append')

# num = torch.Tensor(list(range(0,10)))
# num = num.view(-1,1)
# num = torch.cat((num,num),dim=1)

# plt = vis.line(Y=torch.randn(10,2), X = num)

# plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))

# plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))


# plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))

import visdom
vis = visdom.Visdom()

def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor''' 
    vis.line(X=num, Y=loss_value, win = loss_plot, update='append' )

plt = vis.line(Y=torch.Tensor(1).zero_(), 
        opts=dict(
            # fillarea=True,
            # showlegend=False,
            width=1920,
            # height=800,
            # xlabel='Time',
            # ylabel='Volume',
            # ytype='log',
            title='Test',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        )
    )

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))

vis.close(env="main")