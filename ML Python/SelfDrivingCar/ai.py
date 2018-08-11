# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean

# Creating the architecture of the Neural Network

class Network():

    def __init__(self, input_size, nb_action):        
        self.input_size = input_size
        self.hidden_size = 30
        self.output_size = nb_action

        NN = input_data(shape=[None, self.input_size, 1], name='input')

        NN = fully_connected(NN, self.hidden_size, activation='relu')        
        NN = fully_connected(NN, self.output_size, activation='sigmoid')

        NN = regression(NN, optimizer='adam', learning_rate=0.001, loss='mean_square', name='output')

        self.model = tflearn.DNN(NN, tensorboard_dir='log')
    
    def forward(self, state, batch_size = 1):
        state_reshape = np.reshape(state, (batch_size, self.input_size, -1))        
        q_values = self.model.predict(state_reshape)
        return q_values

    def learn(self, input, target):
        input_reshape = np.reshape(input, (100, 5, -1))
        target_reshape = np.reshape(target, (100, 3))                    
        self.model.fit(input_reshape, target_reshape, n_epoch=1, run_id='self_driving_car_test')        
            
# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self, batch_size):
        samples = list(zip(*random.sample(self.memory, batch_size)))
        return samples

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.network = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)        
        self.last_state = np.array([0., 0., 0., 0., 0.])
        self.last_action = 0
        self.last_reward = 0
        self.batch_size = 100
        self.reward_window_size = 1000
    
    def select_action(self, state):
        probs = self.network.forward(state) # TEMPERATURE MAYBE            
        action = np.argmax(probs)        
        #print(probs, action)
        return action
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.network.forward(batch_state, self.batch_size)
        next_outputs = self.network.forward(batch_next_state, self.batch_size)                                

        target = []        
        for i in range(len(outputs)):                
            best_action = np.argmax(next_outputs[i])
            outputs[i][batch_action[i]] = self.gamma * next_outputs[i][best_action] + batch_reward[i]
            
            outputs_sum = np.sum(outputs[i])
            t = []
            for o in outputs[i]:
                t.append(o / outputs_sum)
            target.append(t) 
        
        self.network.learn(batch_state, target)

    def update(self, reward, new_signal):        
        new_state = np.array(new_signal)        
        self.memory.push((self.last_state, new_state, self.last_action, self.last_reward))
        action = self.select_action(new_state)                
        if len(self.memory.memory) > self.batch_size:                       
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.batch_size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > self.reward_window_size:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)
    
    def save(self):
        # torch.save({'state_dict': self.network.state_dict(),
        #             'optimizer' : self.optimizer.state_dict(),
        #            }, 'last_brain.pth')
        print('not ok')
    
    def load(self):
        # if os.path.isfile('last_brain.pth'):
        #     print("=> loading checkpoint... ")
        #     checkpoint = torch.load('last_brain.pth')
        #     self.network.load_state_dict(checkpoint['state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        #     print("done !")
        # else:
        #     print("no checkpoint found...")
        print('not ok')