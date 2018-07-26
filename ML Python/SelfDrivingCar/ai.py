# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Creating the architecture of the Neural Network

class Network():
    
    def __init__(self, input_size, nb_action):        
        self.input_size = input_size
        self.hidden_size = 30
        self.output_size = nb_action

        self.neural_network = Sequential()
        
        self.neural_network.add(Dense(output_dim = self.hidden_size, init = 'uniform',  activation = 'relu', input_dim = self.input_size))
        self.neural_network.add(Dense(output_dim = self.output_size, init = 'uniform',  activation = 'softmax'))
        #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.neural_network.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    def forward(self, state):            
        q_values = self.neural_network.predict(state)                
        return q_values

    def learn(self, input, target):
        self.neural_network.fit(input, target)

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
        # array([[ 1.,  1.,  1., -0.88716101,  0.88716101]])
        self.last_state = np.expand_dims([0., 0., 0., 0., 0.], axis=0)
        self.last_action = 0
        self.last_reward = 0
        self.batch_size = 100
        self.reward_window_size = 1000
    
    def select_action(self, state):
        probs = self.network.forward(state) # TEMPERATURE MAYBE
        action = np.argmax(probs)        
        return action
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action, batch_size):                        
        batch_next_state_reshape = np.reshape(batch_next_state, (batch_size, self.network.input_size))
        next_outputs = self.network.forward(batch_next_state_reshape)
        
        target = []
        for i in range(len(next_outputs)):        
            t = self.gamma * next_outputs[i] + batch_reward[i]
            target.append(t)        
       
        print(next_outputs)
        print(target)

        binary_target = []
        for t in target:
            bt = [0., 0., 0.]
            i = np.argmax(t.argmax)
            bt[i] = 1
            binary_target.append(bt)        
        
        batch_state_reshape = np.reshape(batch_state, (batch_size, self.network.input_size))
        binary_target_reshape = np.reshape(binary_target, (batch_size, self.network.output_size))        
        self.network.learn(batch_state_reshape, binary_target_reshape)

    def update(self, reward, new_signal):
        new_state = np.expand_dims(new_signal, axis=0)     
        self.memory.push((self.last_state, new_state, self.last_action, self.last_reward))
        action = self.select_action(new_state)
        if len(self.memory.memory) > self.batch_size:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.batch_size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action, self.batch_size)
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