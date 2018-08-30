import numpy as np
import tensorflow as tf

class Agent:

    def __init__(self, actions):        

        self.alpha = 0.1
        self.gamma = 0.9
        LR = 0.0001

        input_size = 1
        h1_size = 4
        h2_size = 10
        self.output_size = actions
        
        with tf.name_scope('input'):
            self.input = tf.placeholder(tf.float32, [None, input_size], name='input')
            self.target = tf.placeholder(tf.float32, [None, self.output_size], name='labels')
        
        with tf.name_scope('h1'):
            wh1 = tf.Variable(tf.random_normal([input_size, h1_size], mean=0, stddev=0.01), name='wh1')
            # tf.summary.histogram('wh1', wh1)
            b1 = tf.Variable(tf.zeros([h1_size]), name='bh1')
            # tf.summary.histogram('b1', b1)

            h1 = tf.add(tf.matmul(self.input, wh1), b1, name='linear_transformation')
            h1 = tf.nn.relu(h1, name='relu')

        with tf.name_scope('h2'):
            wh2 = tf.Variable(tf.random_normal([input_size, h2_size], mean=0, stddev=0.01), name='wh2')
            # tf.summary.histogram('wh2', wh2)
            b2 = tf.Variable(tf.zeros([h2_size]), name='bh2')
            # tf.summary.histogram('b2', b2)

            h2 = tf.add(tf.matmul(self.input, wh2), b2, name='linear_transformation')
            h2 = tf.nn.relu(h2, name='relu')
        
        with tf.name_scope('output'):
            wo = tf.Variable(tf.random_normal([h2_size, self.output_size], mean=0, stddev=0.01), name='wo')
            # tf.summary.histogram('wo', wo)
            bo = tf.Variable(tf.zeros([self.output_size]), name='bo')
            # tf.summary.histogram('bo', bo)
            
            self.output = tf.add(tf.matmul(h2, wo), bo, name='linear_transformation')
            self.output = tf.nn.sigmoid(self.output, name='sigmoid')
            # self.output = tf.nn.softmax(self.output, name='softmax')
            # tf.summary.histogram('output', self.output)

        with tf.name_scope('cost'):      
            # MSE      
            error = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.target), 2))
            self.loss = tf.reduce_mean(error)
            # tf.summary.scalar('cost', self.loss)

            # CATEGORICAL CROSSENTROPY
            # self.loss = tf.reduce_mean(tf.negative(tf.reduce_sum(tf.multiply(self.target, tf.log(self.output)))))

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(self.loss)

        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()        
        self.sess.run(tf.global_variables_initializer())        

        # self.merged = tf.summary.merge_all()
        # self.writter = tf.summary.FileWriter('C:/tmp/logs/train', self.sess.graph)        

    def feedforward(self, s):
        actions = self.sess.run([ self.output ], feed_dict={ self.input: [[ s ]] })        

        return actions

    def play(self, s, verbose = False):
        actions = self.feedforward(s)
        a = np.argmax(actions)

        if verbose:
            print(actions, a)

        return a

    def learn(self, s, a, r, s_, verbose = False):
        output_s = self.feedforward(s)
        output_s_ = self.feedforward(s_)

        if verbose:
            print(s, r)
            print(output_s[0][0], a)

        # new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        a_s_ = np.argmax(output_s_)
        # r + output_s_[0][0][a_s_] * self.gamma
        output_s[0][0][a] = (1 - self.alpha) * output_s[0][0][a] + self.alpha * (r + self.gamma * output_s_[0][0][a_s_])

        self.sess.run([ self.optimizer ], feed_dict={ self.input: [[ s ]], self.target: [ output_s[0][0] ] })

        if verbose:
            print(output_s[0][0])
    