import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from model import Model

class Trainer:

    def __init__(self, feed_size, output_size, fc1_units, fc2_units, lr, checkpoint='./checkpoint.pth'):

        self.model = Model(feed_size, output_size, fc1_units, fc2_units)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.checkpoint = checkpoint

        self.model.load(self.checkpoint)

        self.classifier = LogisticRegression(random_state=0)
        
    def train_with_logistic_regression(self, feed, target):
        # subsets        
        train_feed, test_feed, train_target, test_target = tts(feed, target, test_size=0.1, random_state=0)

        print('Learning: Logistic Regression')

        self.classifier.fit(train_feed, train_target)

        print('\nEnd')
        print('')

        self._test_with_logistic_regression(test_feed, test_target)
    
    def _test_with_logistic_regression(self, feed, target):
        print('Checking accuracy...')
        print('')

        test_predictions = self.classifier.predict(feed)

        test_target = target.astype(np.uint8)
        test_predictions = np.asarray( [ np.round(x) for x in test_predictions ] )    

        cm = confusion_matrix(test_target, test_predictions)

        diagonal_sum = cm.trace()
        all_elements = cm.sum()
        accuracy = diagonal_sum / all_elements
        print('Confusion matrix')
        print(cm)
        print('')
        print('Accuracy: {:.2f}'.format(accuracy))

        print('')
        print('End')
        
    
    def train_with_nn(self, feed, target, batch_size=512, epochs=1):

        # subsets
        train_feed, test_feed, train_target, test_target = tts(feed, target, test_size=0.1, random_state=0)

        print('Learning: Neural Model')

        self.classifier.fit(train_feed, train_target)

        criterion = nn.BCELoss()
        for epoch in range(epochs):
            batch = BatchSampler( SubsetRandomSampler( range(train_feed.shape[0]) ), batch_size, drop_last=False)

            batch_count = 0
            for batch_indices in batch:
                batch_indices = torch.tensor(batch_indices).long()      
                batch_count += 1

                feeds = train_feed[batch_indices]
                targets = train_target[batch_indices]

                # to tensor
                feeds = torch.tensor(feeds).float()
                targets = torch.tensor(targets.astype(np.uint8)).float()
                
                # forward
                predictions = self.model(feeds)

                # loss            
                loss = criterion(predictions, targets)

                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('\rEpoch: \t{} \tBatch: \t{} \tLoss: \t{:.8f}'.format(epoch+1, batch_count, loss.cpu().data.numpy()), end="")  
                    
        print('\nEnd')
        print('')

        self._test_with_nn(test_feed, test_target)

        self.model.checkpoint(self.checkpoint)

    def _test_with_nn(self, feed, target, batch_size=512):

        print('Checking accuracy...')
        print('')

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


        test_target = target.astype(np.uint8)
        test_predictions = np.concatenate( test_predictions )
        test_predictions = np.asarray( [ np.round(x) for x in test_predictions ] )    

        cm = confusion_matrix(test_target, test_predictions)

        diagonal_sum = cm.trace()
        all_elements = cm.sum()
        accuracy = diagonal_sum / all_elements
        print('Confusion matrix')
        print(cm)
        print('')
        print('Accuracy: {:.2f}'.format(accuracy))

        print('')
        print('End')
            
    