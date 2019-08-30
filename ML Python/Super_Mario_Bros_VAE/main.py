import os
from os import system
import cv2
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Decoder


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# HYPERPARAMETERS
DATA_FILES = './data/frame_resized'

BATCH_SIZE = 32
COMPRESSED_FEATURES_SIZE = 1024
SAMPLES = 8
LR = 1e-3


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
files_img = []
for file in files:
    file_name = os.path.join(DATA_FILES, file) 
    files_img.append(
        np.array( cv2.imread(file_name) ).T / 255
    )

epochs = 500
for epoch in range(epochs):     

    r_files_img = files_img.copy()
    random.shuffle(r_files_img)

    average_loss = []
    step = 0

    while r_files_img:
        
        k = BATCH_SIZE if BATCH_SIZE < len(r_files_img) else len(r_files_img)
        batch_img = random.sample( r_files_img, k=k )

        for file in batch_img:
            r_files_img.remove(file)                

        # encoder
        batch_img = torch.from_numpy( np.array(batch_img) ).float().to(DEVICE) 
        mu, logvar = encoder( batch_img )

        z = reparameterize( mu, logvar )

        # decoder
        decoded = [ decoder( z ) for z in z ]

        # cost function            

        # how well do input x and output recon_x agree?    
        MSE = 0
        for recon_x in decoded:
            MSE += F.mse_loss( recon_x, batch_img )
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
        KLD /= BATCH_SIZE * 256 * 240


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


        loss = MSE + KLD + l2_encoder_reg + l2_decoder_reg

        # backward    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            system('cls')
            print('\r Epoch:{} | Step:{} | Loss: {:.5f}'.format( epoch, step, loss.item() ), end='')

            average_loss.append( loss.item() )

            for i, img in enumerate(decoded[0]):
                cv2.imwrite( 'test/{}.jpg'.format(i), img.squeeze(0).cpu().data.numpy().T * 255 )

        step += 1

    # saving models
    encoder.checkpoint('encoder.pth')
    decoder.checkpoint('decoder.pth')

    # generating visualizations
    tb.add_scalar('average_loss', np.average( average_loss ), epoch )

    tb.add_histogram('encoder_conv1.bias', encoder.conv1.bias, epoch)
    tb.add_histogram('encoder_conv1.weight', encoder.conv1.weight, epoch)
    tb.add_histogram('encoder_conv1.weight.grad', encoder.conv1.weight.grad, epoch)

    tb.add_histogram('encoder_conv2.bias', encoder.conv2.bias, epoch)
    tb.add_histogram('encoder_conv2.weight', encoder.conv2.weight, epoch)
    tb.add_histogram('encoder_conv2.weight.grad', encoder.conv2.weight.grad, epoch)

    tb.add_histogram('encoder_conv3.bias', encoder.conv3.bias, epoch)
    tb.add_histogram('encoder_conv3.weight', encoder.conv3.weight, epoch)
    tb.add_histogram('encoder_conv3.weight.grad', encoder.conv3.weight.grad, epoch)

    tb.add_histogram('encoder_conv4.bias', encoder.conv4.bias, epoch)
    tb.add_histogram('encoder_conv4.weight', encoder.conv4.weight, epoch)
    tb.add_histogram('encoder_conv4.weight.grad', encoder.conv4.weight.grad, epoch)

    tb.add_histogram('encoder_fc11.bias', encoder.fc11.bias, epoch)
    tb.add_histogram('encoder_fc11.weight', encoder.fc11.weight, epoch)
    tb.add_histogram('encoder_fc11.weight.grad', encoder.fc11.weight.grad, epoch)

    tb.add_histogram('encoder_fc12.bias', encoder.fc12.bias, epoch)
    tb.add_histogram('encoder_fc12.weight', encoder.fc12.weight, epoch)
    tb.add_histogram('encoder_fc12.weight.grad', encoder.fc12.weight.grad, epoch)

    tb.add_histogram('encoder_fc21.bias', encoder.fc21.bias, epoch)
    tb.add_histogram('encoder_fc21.weight', encoder.fc21.weight, epoch)
    tb.add_histogram('encoder_fc21.weight.grad', encoder.fc21.weight.grad, epoch)

    tb.add_histogram('encoder_fc22.bias', encoder.fc22.bias, epoch)
    tb.add_histogram('encoder_fc22.weight', encoder.fc22.weight, epoch)
    tb.add_histogram('encoder_fc22.weight.grad', encoder.fc22.weight.grad, epoch)

tb.close()