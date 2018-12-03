import numpy as np
import time, threading

import torch
import torch.nn.functional as F
import torch.optim as optim

class Brain:

    def __init__(self, device, model, learning_rate, gamma, min_batch):
        self.train_queue = [ [], [], [], [], [] ]	# state, action, reward, next_state, done
        self.lock_queue = threading.Lock()

        self.DEVICE = device

        self.model = model
        self.LEARNING_RATE = learning_rate
        self.GAMMA = gamma

        self.MIN_BATCH = min_batch      

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():
            act, crit = self.model(state)
            action = act.cpu().data.numpy()
            critic = crit.cpu().data.numpy()
        self.model.train()

        action = [ np.clip(p, -1, 1) for p in action ]

        return action, critic

    def experience_push(self, state, action, reward, next_state, done):
        with self.lock_queue:
            self.train_queue[0].append(state)
            self.train_queue[1].append(action)
            self.train_queue[2].append(reward)
            self.train_queue[3].append(next_state)
            self.train_queue[4].append(done)			

    def learn(self):
        if len(self.train_queue[0]) < self.MIN_BATCH:
            time.sleep(0)	# yield
            return
            
        with self.lock_queue:
            if len(self.train_queue[0]) < self.MIN_BATCH:	# more thread could have passed without lock
                return 									    # we can't yield inside lock
                
            state, action, reward, next_state, done = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]
            
        state = np.vstack(state)
        action = np.vstack(action)
        reward = np.vstack(reward)
        next_state = np.vstack(next_state)
        done = np.vstack(done)
        
        if len(state) > 5*self.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(state))


        action_expected, critic_expected = self.model(state)
        action_next, critic_next = self.model(next_state)

        # CRITIC LOSS
        critic_target = reward + (self.GAMMA * critic_next * (1 - done))
        critic_loss = F.mse_loss(critic_expected, critic_target)

        # ACTOR LOSS
        actor_loss = -critic_expected.mean()

        # TOTAL LOSS
        total_loss = critic_loss + actor_loss


        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        


        
        # # ---------------------------- update critic ---------------------------- #
        # # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_states)
        # Q_targets_next = self.critic_target(next_states, actions_next)
        # # Compute Q targets for current states (y_i)
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # # Compute critic loss
        # Q_expected = self.critic_local(states, actions)
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # # Minimize the loss
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # # ---------------------------- update actor ---------------------------- #
        # # Compute actor loss
        # actions_pred = self.actor_local(states)
        # actor_loss = -self.critic_local(states, actions_pred).mean()
        # # Minimize the loss
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()


          
        # _, value = self.predict(next_state)
        # reward = reward + GAMMA_N * value * (1 - done)	# set v to 0 where s_ is terminal state
        
        # s_t, a_t, r_t, minimize = self.graph
        # self.session.run(minimize, feed_dict={s_t: state, a_t: action, r_t: reward})


        # ###################
        # states = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        # actions = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        # rewards = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
        
        # action, value = model(states)
        


        # log_prob = tf.log( tf.reduce_sum(action * actions, axis=1, keep_dims=True) + 1e-10)
        # advantage = rewards - value
        
        # loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
        # loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
        # entropy = LOSS_ENTROPY * tf.reduce_sum(action * tf.log(action + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)
        # loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        
        # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        # minimize = optimizer.minimize(loss_total)
