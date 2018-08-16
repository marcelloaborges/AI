# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# 
# author: Jaromir Janisch, 2016


#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  

import random, math, gym
import numpy as np

#-------------------- BRAIN ---------------------------
import tensorflow as tf

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self._createModel()        
        #saver = tf.train.import_meta_graph('./cartpole-basic.meta')
        #saver.restore(self.sess, tf.train.latest_checkpoint('./'))        

    def _createModel(self):
        with tf.name_scope('input'):
            self.input = tf.placeholder(tf.float32, [None, stateCnt], name='input')
            self.target = tf.placeholder(tf.float32, [None, actionCnt], name='labels')

        with tf.name_scope('h1'):
            wh1 = tf.Variable(tf.random_normal([stateCnt, 64], mean=0, stddev=0.01), name='wh1')            
            b1 = tf.Variable(tf.zeros([64]), name='bh1')            

            h1 = tf.add(tf.matmul(self.input, wh1), b1, name='relu')
            h1 = tf.nn.relu(h1, name='relu')  

        with tf.name_scope('output'):
            wo = tf.Variable(tf.random_normal([64, actionCnt], mean=0, stddev=0.01), name='wo')            
            bo = tf.Variable(tf.zeros([actionCnt]), name='bo')
            
            self.output = tf.add(tf.matmul(h1, wo), bo, name='linear')

        with tf.name_scope('cost'):      
            # MSE      
            error = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.target), 2))
            self.loss = tf.reduce_mean(error)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0025).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, y, epoch=1, verbose=0):                
        self.sess.run([ self.optimizer ],  feed_dict={ self.input: x, self.target: y })   

    def predict(self, s):        
        return self.sess.run([ self.output ], feed_dict={ self.input: s })[0]

    def predictOne(self, s):        
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])        

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:                
                t[a] = r + GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem

        gym.envs.register(
            id='CartPoleExtraLong-v0',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            max_episode_steps=1000,
            reward_threshold=950.0
        )
        self.env = gym.make('CartPoleExtraLong-v0')

    def run(self, agent):
        s = self.env.reset()
        R = 0 

        while True:            
            self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

try:
    while True:
        env.run(agent)
finally:    
    saver = tf.train.Saver()
    #saver.save(agent.brain.sess, './cartpole-basic')