import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import random

from PIL import Image
from torchvision import models, transforms

class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()

        self.conv1 = nn.Conv2d( in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.mp1   = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d( in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.mp2   = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d( in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.mp3   = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d( in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(32)
        self.mp4   = nn.MaxPool2d(2)        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mp3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.mp4(x)

        x = x.view( x.size(0), -1 )

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
                
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

# HYPERPARAMETERS

LR = 1e-3
N_CLASS = 60 # NUMBER OF DIFFERENT CLASSES IN EACH BATCH

DATA_FOLDER = './data/'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODEL

model = EncoderModel().to(DEVICE)
# IN CASE YOU WANT TO LOAD ANY PREVIOUS TRAINING UNCOMMENT THIS LINE
# model.load('./MODEL.pth', DEVICE)

optimizer = optim.Adam( model.parameters(), lr=LR )
scaler = GradScaler()

imgToTensor = transforms.ToTensor()
tensorToImg = transforms.ToPILImage()


# SAMPLES AND BATCHES

samples = []   

for _, alphas, _ in os.walk( DATA_FOLDER ):
    for alpha in alphas:
        base_alphas = "{}/{}".format( DATA_FOLDER, alpha )
    
        for _, chars, _ in os.walk( base_alphas ):
            for ch in chars:
                base_chars = "{}/{}".format( base_alphas, ch )

                for _, _, imgs in os.walk( base_chars ):

                    # ALL CHARS HAVE 20 EXAMPLES                    
                    supp_imgs = imgs[:5]
                    query_imgs = imgs[5:10]
                    test_imgs = imgs[10:]
                    
                    samples.append( 
                        {
                            'class': '{}-{}'.format( alpha, ch ),
                            'base_folder': base_chars,
                            'support': supp_imgs,
                            'query': query_imgs,
                            'test': test_imgs
                        }
                    )    

def train_eval():    
    
    idxs = BatchSampler( SubsetRandomSampler( range(0, len(samples)) ), N_CLASS, drop_last=True )    

    print('Loading batch imgs')

    batches = []
    for ind in idxs:        
        selected_samples = [ samples[i] for i in ind ]
       
        batch = []
        for i, sample in enumerate(selected_samples):
            
            def load_imgs(sample, prop):
                imgs = []
                for supp in sample[prop]:
                    file_name = '{}/{}'.format( sample['base_folder'], supp )
                    img_pil = Image.open(file_name)
                    img_pil.thumbnail( (28, 28), Image.ANTIALIAS )
                    img = np.array(img_pil)

                    imgs.append( img )

                return imgs

            xs = load_imgs( sample, 'support' )
            xq = load_imgs( sample, 'query' )
            xt = load_imgs( sample, 'test' )

            class_tensor = {
                'idx': i,
                'char': sample['class'],
                'tensor': {
                    'xs': xs,
                    'xq': xq,
                    'xt': xt
                }
            }

            batch.append(class_tensor)
        
        batches.append(batch)

    print('Starting training')

    for epoch in range(100):   

        with autocast():       

            for i, batch in enumerate(batches):                    

                model.train()         

                xs = []
                xq = []
                
                xs.append( [ s['tensor']['xs'] for s in batch ] )
                xq.append( [ q['tensor']['xs'] for q in batch ] )

                xs = torch.from_numpy( np.array( xs ) ).float().to(DEVICE)
                xq = torch.from_numpy( np.array( xq ) ).float().to(DEVICE)

                xs = xs/255.
                xq = xq/255.

                xs = xs.squeeze(0).unsqueeze(2)
                xq = xq.squeeze(0).unsqueeze(2)
                
                # LEARN
                n_class = xs.size(0)
                assert xq.size(0) == n_class
                n_support = xs.size(1)
                n_query = xq.size(1)

                target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(DEVICE)

                x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                            xq.view(n_class * n_query, *xq.size()[2:])], 0)

                z = model(x)
                z_dim = z.size(-1)

                z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
                zq = z[n_class*n_support:]

                dists = euclidean_dist(zq, z_proto)

                log_p_y = F.log_softmax(-dists, dim=0).view(n_class, n_query, -1)

                train_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

                _, y_hat = log_p_y.max(2)
                train_acc = torch.eq(y_hat, target_inds.squeeze()).float().mean()
                

                optimizer.zero_grad()
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss = train_loss.cpu().data.item()
                train_acc = train_acc.cpu().data.item()                                

                print('\rEpoch:{} batch:{}/{}'.format(epoch+1, i+1, len(batches)), end='')

            print('')

            # NOT SAVING THE MODEL BECAUSE THE FUNCTION BUILDS RANDOM SAMPLES FOR EVERY REQUEST
            # IF YOU NEED TO SAVE THE TRAINING USE THIS FUNCTION
            # model.checkpoint('./MODEL.pth')

            # TEST
            model.eval()

            test_batch = random.sample( batches, k=1 )[0]

            xs = []
            xt = []
            
            xs.append( [ s['tensor']['xs'] for s in test_batch ] )
            xt.append( [ t['tensor']['xt'] for t in test_batch ] )

            xs = torch.from_numpy( np.array( xs ) ).float().to(DEVICE)
            xt = torch.from_numpy( np.array( xt ) ).float().to(DEVICE)

            xs = xs/255.
            xt = xt/255.

            xs = xs.squeeze(0).unsqueeze(2)
            xt = xt.squeeze(0).unsqueeze(2)

            n_class = xs.size(0)
            assert xt.size(0) == n_class
            n_support = xs.size(1)
            n_query = xt.size(1)

            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(DEVICE)

            x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                        xt.view(n_class * n_query, *xt.size()[2:])], 0)

            with torch.no_grad():
                z = model(x)
            
            z_dim = z.size(-1)

            z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
            zq = z[n_class*n_support:]

            dists = euclidean_dist(zq, z_proto)

            log_p_y = F.log_softmax(-dists, dim=0).view(n_class, n_query, -1)

            test_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

            _, y_hat = log_p_y.max(2)
            test_acc = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            
            test_loss = test_loss.cpu().data.item()
            test_acc = test_acc.cpu().data.item()
                    
            print('Train: Loss: {:.5f} Acc: {:.5f}'.format(train_loss, train_acc))
            print('Test:  Loss: {:.5f} Acc: {:.5f}'.format(test_loss, test_acc))


            print('')
            test_inx = np.random.choice( 60, 5 )
            for i in test_inx:                
                pred_idx = y_hat[i].cpu().data.numpy()
                score = 0
                for idx in pred_idx:
                    if idx == i:
                        score += 1

                print('Target: "{}" | {}/10'.format( test_batch[i]['char'], score ) )
            print('')



train_eval()