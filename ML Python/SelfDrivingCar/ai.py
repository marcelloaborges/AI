# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import tensorflow as tf

from statistics import median, mean

# Creating the architecture of the Neural Network

class Network():

    def __init__(self, input_size, nb_action):        
        self.input_size = input_size
        self.hidden_size = 30
        self.output_size = nb_action

        with tf.name_scope('input'):
            self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
            self.target = tf.placeholder(tf.float32, [None, self.output_size], name='labels')
        
        with tf.name_scope('h1'):
            wh1 = tf.Variable(tf.random_normal([input_size, self.hidden_size], mean=0, stddev=0.01), name='wh1')            
            b1 = tf.Variable(tf.zeros([self.hidden_size]), name='bh1')

            h1 = tf.add(tf.matmul(self.input, wh1), b1, name='linear_transformation')
            h1 = tf.nn.relu(h1, name='relu')

        with tf.name_scope('output'):
            wo = tf.Variable(tf.random_normal([self.hidden_size, self.output_size], mean=0, stddev=0.01), name='wo')
            bo = tf.Variable(tf.zeros([self.output_size]), name='bo')
            
            self.output = tf.add(tf.matmul(h1, wo), bo, name='linear_transformation')
            self.output = tf.nn.sigmoid(self.output, name='sigmoid')

        with tf.name_scope('cost'):      
            # MSE      
            error = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.target), 2))
            self.loss = tf.reduce_mean(error)

            # CATEGORICAL CROSSENTROPY
            # self.loss = tf.reduce_mean(tf.negative(tf.reduce_sum(tf.multiply(self.target, tf.log(self.output)))))

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        
        self.sess = tf.Session()        
        self.sess.run(tf.global_variables_initializer())        
    
    def forward(self, state):        
        q_values = self.sess.run([ self.output ], feed_dict={ self.input: state })        
        return q_values

    def learn(self, input, target):      
        self.sess.run([ self.optimizer ],  feed_dict={ self.input: input, self.target: target })
            
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
        self.last_state = np.array([[0., 0., 0., 0., 0.]])
        self.last_action = 0
        self.last_reward = 0
        self.batch_size = 100
        self.reward_window_size = 1000
    
    def select_action(self, state):        
        probs = self.network.forward(state) # TEMPERATURE MAYBE            
        action = np.argmax(probs)
        print(probs, action)
        return action
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):                
        r_batch_state = []
        r_batch_next_state = []
        for i in range(self.batch_size):
            r_batch_state.append(batch_state[i][0])        
            r_batch_next_state.append(batch_next_state[i][0])
                
        outputs = self.network.forward(r_batch_state)
        next_outputs = self.network.forward(r_batch_next_state)        
                
        target = []
        for i in range(self.batch_size):            
            best_action = np.argmax(next_outputs[0][i])
            outputs[0][i][batch_action[i]] = self.gamma * next_outputs[0][i][best_action] + batch_reward[i]
            
            target.append(outputs[0][i])
            # t = [0., 0., 0.]                        
            # target_best_action = np.argmax(outputs[0][i])            
            # t[target_best_action] = 1.
            # target.append(t)
                
        self.network.learn(r_batch_state, target)

    def update(self, reward, new_signal):        
        new_state = np.array([new_signal])        
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