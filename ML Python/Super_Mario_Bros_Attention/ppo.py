import numpy as np

from collections import deque
import pickle

from PIL import Image

import torch
from torchvision import models, transforms

class PPO:

    def __init__(self, 
        device,
        env, 
        agent,        
        action_size, 
        img_h, img_w, 
        seq_len
        ):
        
        self.DEVICE = device

        self.env = env
        self.agent = agent
        
        self.action_size = action_size

        self.SEQ_LEN = seq_len
        self.img_h = img_h
        self.img_w = img_w

        self.imgToTensor = transforms.ToTensor()
        self.tensorToImg = transforms.ToPILImage()

    def train(self, n_episodes, height_pixel_cut=15):
        
        avg_loss = (0.0, 0.0)
        for episode in range(n_episodes):
            
            total_reward = 0            

            state = self.env.reset()

            state = Image.fromarray(state).resize( ( self.img_w, self.img_h ) )
            state = transforms.functional.to_grayscale(state)
            # self.tensorToImg( self.imgToTensor(state) ).save('check.jpg')
            # self.tensorToImg( self.imgToTensor(state)[:,height_pixel_cut:,:] ).save('check1.jpg')
            state = self.imgToTensor(state)[:,height_pixel_cut:,:].cpu().data.numpy() 

            seq_state = deque( maxlen=self.SEQ_LEN )
            for _ in range(self.SEQ_LEN):
                seq_state.append( np.zeros( state.shape ) )

            seq_state.append( state )

            while True:        
                
                action, log_prob = self.agent.act( seq_state )

                next_state, reward, done, _ = self.env.step(action)

                next_state = Image.fromarray(next_state).resize( ( self.img_w, self.img_h ) )
                next_state = transforms.functional.to_grayscale(next_state)
                # tensorToImg( imgToTensor(state)).save('check2.jpg')
                next_state = self.imgToTensor(next_state)[:,height_pixel_cut:,:].cpu().data.numpy()                

                loss = self.agent.step( seq_state, action, log_prob, reward )

                self.env.render()

                total_reward += reward

                seq_state.append( next_state )

                if done:
                    # self.agent.checkpoint()
                    break                

                avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
                
                print('\rE: {} TR: {} R: {} L: {:.5f}'.format( episode + 1, total_reward, reward, avg_loss[0]/avg_loss[1] ), end='')

        self.env.close()