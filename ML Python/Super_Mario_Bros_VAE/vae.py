import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128


class VAE(nn.Module):
    def __init__(self, channels=1, img_rows=256, img_cols=240):
        super(VAE, self).__init__()
        
        # enconder
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.state_size = 32 * 4 * 4
        self.features_output = 64

        self.fc11 = nn.Linear(in_features=self.state_size, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=self.features_output)

        self.fc21 = nn.Linear(in_features=self.state_size, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=self.features_output)        


        # decoder
        self.fc1 = nn.Linear(in_features=self.features_output, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.state_size)        

        self.conv_t1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv_t2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv_t3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv_t4 = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=3, stride=2, padding=1)


    def encode(self, state):

        x = F.elu( self.conv1(state) )
        x = F.elu( self.conv2(x) )
        x = F.elu( self.conv3(x) )
        x = F.elu( self.conv4(x) )

        # Flatten
        x = x.view( -1, self.state_size )

        mu_z = F.elu(self.fc11(x))
        mu_z = self.fc12(mu_z)

        logvar_z = F.elu(self.fc21(x))
        logvar_z = self.fc22(logvar_z)

        return mu_z, logvar_z           

    def decode(self, features):

        x = F.elu(self.fc1(features))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 32, 4, 4)
        x = F.relu(self.conv_t1(x))
        x = F.relu(self.conv_t2(x))
        x = F.relu(self.conv_t3(x))
        x = F.sigmoid(self.conv_t4(x))

        return x.view(-1, 60 * 64)
    
    def forward(self, state):
        # check if this is dealing with batches
        state = torch.from_numpy( state ).float().unsqueeze(0).to(DEVICE)

        mu, logvar = self.encode(state)
                
        # multiply log variance with 0.5, then in-place exponent
        # yielding the standard deviation
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)

        if self.training:
            return self.decode(z), mu, logvar
        else:
            return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # how well do input x and output recon_x agree?

        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, 784))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= BATCH_SIZE * 64 * 60


        return BCE + KLD


model = VAE().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

import glob 
import cv2
import sys

data_files = './data'
resize_dim = ( 240, 256 )


def train():
    # toggle model to train mode
    model.train()
    train_loss = 0    
            
    for img in glob.glob(data_files+'/*.*'):        
        var_img = cv2.imread(img)

        resized_img = cv2.resize( var_img, resize_dim, interpolation = cv2.INTER_AREA )

        cv2.imshow(str(img) , resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = model.loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    
train()