import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import confusion_matrix

from model import EmbeddingNet

class Trainer:

    def __init__(self, device, n_users, n_movies, n_factors, hidden, embedding_dropout, dropouts, lr, weight_decay, checkpoint='./checkpoint.pth'):

        self.DEVICE = device
        self.CHECKPOINT = checkpoint

        self.model = EmbeddingNet(
            n_users=n_users, 
            n_movies=n_movies, 
            n_factors=150, 
            hidden=[500, 500, 500], 
            embedding_dropout=0.05,
            dropouts=[0.5, 0.5, 0.25]).to(self.DEVICE)

        self.model.load(self.CHECKPOINT)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train(self, feed_train, feed_test, feed_target, feed_val, minmax, batch_size=512, epochs=1):
        print('Learning...')

        for epoch in range(epochs):

            batches = BatchSampler( SubsetRandomSampler( range(feed_train.shape[0]) ), batch_size, drop_last=False)
            
            batch_count = 0
            for batch_indices in batches:
                batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)
                batch_count += 1

                feed_user = feed_train[:, 0][batch_indices]
                feed_movie = feed_train[:, 1][batch_indices]
                target = feed_target[batch_indices]

                feed_user = torch.tensor(feed_user).long().to(self.DEVICE)
                feed_movie = torch.tensor(feed_movie).long().to(self.DEVICE)
                target = torch.tensor(target).float().to(self.DEVICE)

                predictions = self.model(feed_user, feed_movie, minmax).squeeze(1)

                loss = F.smooth_l1_loss(predictions, target)

                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('\rEpoch: \t{} \tBatch: \t{} \tLoss: \t{:.8f}'.format(epoch+1, batch_count, loss.cpu().data.numpy()), end="")  

            torch.save(self.model.state_dict(), self.CHECKPOINT)

        print('\nEnd')
        print('')

        self._test(feed_test, feed_val, minmax, batch_size)

    def _test(self, feed_test, feed_val, minmax, batch_size=512):         
        batches = BatchSampler( SubsetRandomSampler( range(feed_test.shape[0]) ), batch_size, drop_last=False)

        print('Checking test loss...')
        print('')

        self.model.eval()
        with torch.no_grad():

            ground_truth = []
            predictions = []
            for batch_indices in batches:
                batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

                feed_user = feed_test[:, 0][batch_indices]
                feed_movie = feed_test[:, 1][batch_indices]

                feed_user = torch.tensor(feed_user).long().to(self.DEVICE)
                feed_movie = torch.tensor(feed_movie).long().to(self.DEVICE)
                
                predictions.append( self.model(feed_user, feed_movie, minmax).squeeze(1).cpu().data.numpy() )
                ground_truth.append( feed_val[batch_indices] )

            ground_truth = np.concatenate( (ground_truth), axis=0 ).ravel()
            predictions = np.concatenate( (predictions), axis=0 ).ravel()

            final_loss = np.sqrt(np.mean( ( predictions - ground_truth ) ** 2 ) )
            print(f'Final RMSE: {final_loss:.4f}')    

        print('')
        print('End')
