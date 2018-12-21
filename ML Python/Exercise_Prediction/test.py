import numpy as np

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from model import Model

class Tester:

    def __init__(self, feed_size, output_size, fc1_units, fc2_units, checkpoint='./checkpoint.pth'):
        self.model = Model(feed_size, output_size, fc1_units, fc2_units)

        self.checkpoint = checkpoint
        
        self.model.load(self.checkpoint)

    def test(self, feed, batch_size=512):
        print('Testing...')

        batch = BatchSampler( SubsetRandomSampler( range(feed.shape[0]) ), batch_size, drop_last=False)

        test_predictions = []
        for batch_indices in batch:
            batch_indices = torch.tensor(batch_indices).long()

            feeds = feed[batch_indices]

            # to tensor
            feeds = torch.tensor(feeds).float()        
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(feeds).cpu().data.numpy()
            
            test_predictions.append( predictions )

            # for i in range( len(predictions) ):
            #     print('Target: \t{} \t Predict: \t{} {}'.format( target_set[i].squeeze(0), False if predictions[i] < 0.5 else True, predictions[i] ))  
        
        test_predictions = np.concatenate( test_predictions )

        print('\nEnd')
        print('')

        return test_predictions