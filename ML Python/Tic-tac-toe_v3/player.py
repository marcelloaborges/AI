import numpy as np
import tensorflow as tf

class Player:

    def __init__(self, action):        
        self.action = action
        self.memory = Memory()
        self.count = 0

        self.gamma = 0.9
        LR = 0.001

        input_size = 9
        h1_size = 60
        h2_size = 30
        output_size = 9
        
        with tf.name_scope('input'):
            self.input = tf.placeholder(tf.float32, [None, input_size], name='input')
            self.target = tf.placeholder(tf.float32, [None, output_size], name='labels')
        
        with tf.name_scope('h1'):
            wh1 = tf.Variable(tf.random_normal([input_size, h1_size], mean=0, stddev=0.01), name='wh1')
            tf.summary.histogram('wh1', wh1)
            b1 = tf.Variable(tf.zeros([h1_size]), name='bh1')
            tf.summary.histogram('b1', b1)

            h1 = tf.add(tf.matmul(self.input, wh1), b1, name='linear_transformation')
            h1 = tf.nn.relu(h1, name='relu')

        with tf.name_scope('h2'):
            wh2 = tf.Variable(tf.random_normal([h1_size, h2_size], mean=0, stddev=0.01), name='wh2')
            tf.summary.histogram('wh2', wh2)
            b2 = tf.Variable(tf.zeros([h2_size]), name='bh2')
            tf.summary.histogram('b2', b2)

            h2 = tf.add(tf.matmul(h1, wh2), b2, name='linear_transformation')
            h2 = tf.nn.relu(h2, name='relu')

        with tf.name_scope('output'):
            wo = tf.Variable(tf.random_normal([h2_size, output_size], mean=0, stddev=0.01), name='wo')
            tf.summary.histogram('wo', wo)
            bo = tf.Variable(tf.zeros([output_size]), name='bo')
            tf.summary.histogram('bo', bo)
            
            self.output = tf.add(tf.matmul(h2, wo), bo, name='linear_transformation')
            self.output = tf.nn.sigmoid(self.output, name='sigmoid')

        with tf.name_scope('cost'):      
            # MSE      
            error = tf.reduce_sum(tf.pow(tf.subtract(self.output, self.target), 2))
            self.loss = tf.reduce_mean(error)
            tf.summary.scalar('cost', self.loss)

            # CATEGORICAL CROSSENTROPY
            #self.loss = tf.reduce_mean(tf.negative(tf.reduce_sum(tf.multiply(self.target, tf.log(self.output)))))

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.loss)

        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        self.writter = tf.summary.FileWriter('C:/tmp/logs/train', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())        

        # # para salvar o modelo treinado
        # saver = tf.train.Saver()
                    
        # print(input)
        # print(scores)

    def forward(self, state):                    
        output = self.sess.run([ self.output ], feed_dict={ self.input: np.array([state]) })                                
        return output

    def play(self, state):                        
        output = self.forward(state)
        action = np.argmax(output)                
        return action

    def learn(self):                
        merged = tf.summary.merge_all()
        for episode in self.memory.read():
            state = episode[0]
            action = episode[1]
            reward = episode[2]
            next_state = episode[3]

            output = self.forward(state)                
            next_output = self.forward(next_state)

            action_next_output = np.argmax(next_output)               
            new_value_action = self.gamma * next_output[0][action_next_output] + reward        
            output[0][action] = new_value_action        
               
            self.sess.run([ self.optimizer ],  feed_dict={ self.input: np.array([ state ]), self.target: np.array([ output ]) })
            summary, _ = self.sess.run([ merged, self.loss ],  feed_dict={ self.input: np.array([ state ]), self.target: np.array([ output ]) })
            self.writter.add_summary(summary, self.count)
            self.count += 1
    
    def add_memory(self, state, action, reward, new_state):
        self.memory.add(state, action, reward, new_state)

class Memory:
    state = []        
    action = []
    reward = []
    new_state = []

    def add(self, state, action, reward, new_state):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.new_state.append(new_state)

    def read(self):
        episodes = []
        for i in range(len(self.state)):
            episodes.append(self.state[i], self.action[i], self.reward[i], self.new_state[i])
        return episodes

    def reset(self):
        self.state = []        
        self.action = []
        self.reward = []
        self.new_state = []




# p1 = Player('X')
# actions = []

# for i in range(10):
#     reward = 1
#     s = [-1, -1, -1,  # [-1, -1, -1,
#          -1, -1, -1,  # -1,  1, -1,
#          -1, -1, -1]  # -1, -1, -1 ]

#     output, action = p1.action(s)
#     actions.append((output, action))

#     ns = [-1, -1, -1,
#           -1, -1, -1,
#           -1, -1, -1]
#     ns[action] = 1

#     if action != 4:
#         reward = 0
        
#     p1.update(s, action, reward, ns)    

# print(actions)