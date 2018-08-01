# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers

# Creating the architecture of the Neural Network

class Network():
    
    def __init__(self, input_size, nb_action):        
        self.input_size = input_size
        self.hidden_size = 30
        self.output_size = nb_action

        self.neural_network = Sequential()
        
        self.neural_network.add(Dense(output_dim = self.hidden_size, 
            kernel_regularizer = regularizers.l2(0.01),
            activity_regularizer = regularizers.l1(0.01),    
            init = 'uniform',  
            activation = 'relu', 
            input_dim = self.input_size))
        self.neural_network.add(Dense(output_dim = self.output_size, init = 'uniform',  activation = 'softmax'))
        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)        
        self.neural_network.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    def forward(self, state, batch_size):
        state_reshape = np.reshape(state, (batch_size ,self.input_size))
        q_values = self.neural_network.predict(state_reshape)
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
        self.last_state = np.array([0., 0., 0., 0., 0.])
        self.last_action = 0
        self.last_reward = 0
        self.batch_size = 100
        self.reward_window_size = 1000
    
    def select_action(self, state):
        probs = self.network.forward(state, 1) # TEMPERATURE MAYBE        
        action = np.argmax(probs)                        
        print(probs, action)
        return action
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action, batch_size):        
        next_outputs = self.network.forward(batch_next_state, self.batch_size)                        
        target = []
        for i in range(len(next_outputs)):            
            #print(batch_state[i], batch_action[i], batch_reward[i])
            next_outputs[i] = self.gamma * next_outputs[i] + batch_reward[i]            

            t = [0., 0., 0.]
            best_action = np.argmax(next_outputs[i])            
            t[best_action] = 1.            
            target.append(np.array(t))
            #print(np.array(batch_state[i]), np.array(target[i]))
                                        
        self.network.learn(np.array(batch_state), np.array(target))

    def update(self, reward, new_signal):
        print(reward)
        new_state = np.array(new_signal)        
        self.memory.push((self.last_state, new_state, self.last_action, self.last_reward))
        action = self.select_action(new_state)                
        if len(self.memory.memory) > self.batch_size:           
            #print(self.memory.memory)            
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