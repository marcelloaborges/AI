import os
from os import system
import random
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Decoder


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# HYPERPARAMETERS
DATA_FILES = './data/frame_resized'

BATCH_SIZE = 32
COMPRESSED_FEATURES_SIZE = 256
SAMPLES = 4
LR = 1e-3
CICLICAL_B_STEPS = 1000


encoder = Encoder(COMPRESSED_FEATURES_SIZE).to(DEVICE)
decoder = Decoder(COMPRESSED_FEATURES_SIZE).to(DEVICE)
optimizer = optim.Adam( list(encoder.parameters()) + list(decoder.parameters()), lr=LR )

encoder.load('encoder.pth')
decoder.load('decoder.pth')

tb = SummaryWriter('runs')


# multiply log variance with 0.5, then in-place exponent
# yielding the standard deviation
def reparameterize(mu, logvar):
    samples_z = []

    for _ in range(SAMPLES):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        samples_z.append( z )    

    return samples_z


files = os.listdir( DATA_FILES )

B_step = 0
hold_B = False
epochs = 100
step = 0

imgToTensor = transforms.ToTensor()
tensorToImg = transforms.ToPILImage()
for epoch in range(epochs):
    
    r_files = files.copy()
    random.shuffle(r_files)        

    while r_files:
        
        k = BATCH_SIZE if BATCH_SIZE < len(r_files) else len(r_files)
        batch_files = random.sample( r_files, k=k )

        batch_img = []
        for file in batch_files:
            file_name = os.path.join(DATA_FILES, file)             
            img_pil = Image.open(file_name)
            img_tensor = imgToTensor(img_pil).float().to(DEVICE)
            batch_img.append(
                img_tensor
            )

        for file in batch_files:
            r_files.remove(file)                

        # encoder
        batch_img = torch.stack( batch_img )
        mu, logvar = encoder( batch_img )

        zs = reparameterize( mu, logvar )

        # decoder
        decoded = [ decoder( z ) for z in zs ]

        # cost function            

        # how well do input x and output recon_x agree?    
        MSE = 0
        for recon_x in decoded:
            # MSE += F.mse_loss( recon_x.reshape((-1, 3 * 240 * 256)), batch_img.reshape((-1, 3 * 240 * 256)) )
            exp = ( batch_img.reshape((-1, 3, 240 * 256)) - recon_x.reshape((-1, 3, 240 * 256)) ) ** 2
            MSE += ( ( exp ).sum(dim=2) ).mean()
        MSE /= SAMPLES * BATCH_SIZE


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
        KLD /= BATCH_SIZE * COMPRESSED_FEATURES_SIZE


        # L2 regularization
        l2_factor = 1e-6
        
        l2_encoder_reg = None
        for W in encoder.parameters():
            if l2_encoder_reg is None:
                l2_encoder_reg = W.norm(2)
            else:
                l2_encoder_reg = l2_encoder_reg + W.norm(2)

        l2_encoder_reg = l2_encoder_reg * l2_factor

        l2_decoder_reg = None
        for W in decoder.parameters():
            if l2_decoder_reg is None:
                l2_decoder_reg = W.norm(2)
            else:
                l2_decoder_reg = l2_decoder_reg + W.norm(2)

        l2_decoder_reg = l2_decoder_reg * l2_factor


        # ciclical B
        B = B_step / CICLICAL_B_STEPS if not hold_B else 1
        if B_step == CICLICAL_B_STEPS:
            hold_B = not hold_B
            B_step = 0


        # final loss
        loss = MSE + (B * KLD) + l2_encoder_reg + l2_decoder_reg

        # backward    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            system('cls')
            print('\r Epoch:{} | Step:{} | B_Step: {} | B: {:.2f} | Loss: {:.5f}, MSE: {:.5F}, KLD: {:.5F}'.format( epoch, step, B_step, B, loss.item(), MSE.item(), KLD.item() ), end='')

            for i, img in enumerate(decoded[0]):
                tensorToImg(img.cpu()).save('test/{}.jpg'.format(i))

            # generating visualizations        

            # test file
            test_file_name = os.path.join(DATA_FILES, 'frame240.jpg') 
            img_pil = Image.open(test_file_name)

            # encoder
            test_img = imgToTensor(img_pil).float().unsqueeze(0).to(DEVICE)
            mu, logvar = encoder( test_img )

            zs = reparameterize( mu, logvar )

            # decoder
            decoded = decoder( zs[0] )

            # adding tensorboard
            tb.add_scalar('loss', loss.item(), step )
            tb.add_scalar('MSE', MSE.item(), step )
            tb.add_scalar('KLD', KLD.item(), step )
            tb.add_image('encoder_input', test_img.squeeze(0), step)
            tb.add_image('decoder_output', decoded.squeeze(0), step)

            tb.add_histogram('encoder_conv1.bias', encoder.conv1.bias, step)
            tb.add_histogram('encoder_conv1.bias.grad', encoder.conv1.bias.grad, step)
            tb.add_histogram('encoder_conv1.weight', encoder.conv1.weight, step)
            tb.add_histogram('encoder_conv1.weight.grad', encoder.conv1.weight.grad, step)

            tb.add_histogram('decoder.conv1.bias', decoder.conv_t1.bias, step)
            tb.add_histogram('decoder.conv1.bias.grad', decoder.conv_t1.bias.grad, step)
            tb.add_histogram('decoder.conv1.weight', decoder.conv_t1.weight, step)
            tb.add_histogram('decoder.conv1.weight.grad', decoder.conv_t1.weight.grad, step)

        B_step += 1
        step += 1

    # saving models
    encoder.checkpoint('encoder.pth')
    decoder.checkpoint('decoder.pth')    

tb.close()