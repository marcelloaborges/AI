import numpy as np

from collections import deque
import pickle

from PIL import Image

import torch
from torchvision import models, transforms

class DataExtractor:

    def __init__(self, 
        device,
        env,         
        action_size, 
        img_h, img_w, 
        seq_len,
        attention_model, dqn_model,        
        batch_size=256,
        total_exp=256000,
        batch_folder='./'
        ):
        
        self.DEVICE = device

        self.env = env
        
        self.action_size = action_size

        self.SEQ_LEN = seq_len
        self.img_h = img_h
        self.img_w = img_w

        self.imgToTensor = transforms.ToTensor()
        self.tensorToImg = transforms.ToPILImage()

        self.attention_model = attention_model
        self.dqn_model = dqn_model

        self.attention_model.eval()    
        self.dqn_model.eval()

        self.BATCH_FOLDER = batch_folder

        self.memory = []
        self.batch_size = batch_size
        self.total_exp = total_exp
        

    def extract(self, height_pixel_cut=15):
        
        qtd_exp = 0        
        while qtd_exp < self.total_exp:
            state = self.env.reset()

            state = Image.fromarray(state).resize( ( self.img_w, self.img_h ) )
            state = transforms.functional.to_grayscale(state)
            # self.tensorToImg( self.imgToTensor(state) ).save('check.jpg')
            # self.tensorToImg( self.imgToTensor(state)[:,height_pixel_cut:,:] ).save('check1.jpg')
            state = self.imgToTensor(state)[:,height_pixel_cut:,:].cpu().data.numpy() 

            seq_state = deque( maxlen=self.SEQ_LEN )
            seq_action = deque( maxlen=self.SEQ_LEN )
            seq_reward = deque( maxlen=self.SEQ_LEN )
            seq_done = deque( maxlen=self.SEQ_LEN )
            for _ in range(self.SEQ_LEN):
                seq_state.append( np.zeros( state.shape ) )
                seq_action.append( np.zeros( (1, self.action_size) ) )
                seq_reward.append( 0 )
                seq_done.append( False )        

            seq_state.append( state )

            seq_n = 0
            t_steps = self.SEQ_LEN * 40
            for _ in range(t_steps) :

                with torch.no_grad():            
                    t_seq_state = torch.tensor(seq_state).unsqueeze(0).float().to(self.DEVICE)

                    encoded = self.attention_model(t_seq_state)
                    dist = self.dqn_model(encoded[:,-1:])

                action = dist.sample()  
        
                dist = dist.logits.cpu().data.numpy()
                action = action.cpu().data.numpy()[0]

                next_state, reward, done, _ = self.env.step(action)

                next_state = Image.fromarray(next_state).resize( ( self.img_w, self.img_h ) )
                next_state = transforms.functional.to_grayscale(next_state)
                # tensorToImg( imgToTensor(state)).save('check2.jpg')
                next_state = self.imgToTensor(next_state)[:,height_pixel_cut:,:].cpu().data.numpy()
                
                seq_next_state = seq_state.copy()
                seq_next_state.append( next_state )


                seq_action.append( dist )
                seq_reward.append( reward )
                seq_done.append( done )


                e = {
                    "states" : seq_state,
                    "actions" : seq_action,
                    "rewards" : seq_reward,
                    "next_states" : seq_next_state,
                    "dones" : seq_done
                }

                in_memory = False
                for v in self.memory:
                    comparison = v['states'][-1] == e['states'][-1]
                    equals = comparison.all()

                    if equals:
                        in_memory = True
                        break
                    
                if not in_memory:
                    seq_n += 1

                    if seq_n >= self.SEQ_LEN:
                        self.memory.append(e)
                        qtd_exp += 1         
                        seq_n = 0           


                self.env.render()

                seq_state = seq_next_state                              

                print('\r Exp: {}'.format( qtd_exp ), end='')

                if len(self.memory) >= self.batch_size:                    
                    self._save( qtd_exp, self.memory )
                    self.memory = []                    

                if done:
             
                    break  
    
    def _save(self, exp_n, memory):
        checkpoint_file = '{}batch_{}.btch'.format( self.BATCH_FOLDER, exp_n )
        batch_file = open(checkpoint_file, 'wb') 
        pickle.dump( memory, batch_file )
        batch_file.close()