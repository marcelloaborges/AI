import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch.optim as optim

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def kaiming_layer_init(layer, mode='fan_out', nonlinearity='relu'):
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    return layer

def kaiming_weight_init(weight, mode='fan_out', nonlinearity='relu'):
    nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    return weight

class AttentionEncoderModel(nn.Module):

    def __init__(self, seq_len, attention_heads, img_h, img_w, 
        fc1_units=4096, fc2_units=2048, fc3_units=1024, fc4_units=512,
        compressed_features_size=128, device='cpu'):
        super(AttentionEncoderModel, self).__init__()

        self.DEVICE = device        

        self.seq_len = seq_len
        self.attention_heads = attention_heads
        self.img_embedding = 256 
        self.n_attention_blocks = 8
        self.output_size = 256 # base 64 / attention heads must be integer and even        
        self.compressed_features_size = compressed_features_size

        # FLATTEN IMG EMBEDDING
        self.fc1_w, self.fc1_baias =\
            self._generate_w_bias(img_h * img_w, fc1_units)
        self.fc2_w, self.fc2_baias =\
            self._generate_w_bias(fc1_units, fc2_units)
        self.fc3_w, self.fc3_baias =\
            self._generate_w_bias(fc2_units, fc3_units)
        self.fc4_w, self.fc4_baias =\
            self._generate_w_bias(fc3_units, fc4_units)
        self.fc5_w, self.fc5_baias =\
            self._generate_w_bias(fc4_units, self.img_embedding)

        # MLP (CUSTOM CONV1d)
        self.pre_enconding_w, self.pre_encoding_baias =\
            self._generate_w_bias(self.img_embedding, self.output_size)

        # POSITIONAL
        self.positional_w =\
            kaiming_weight_init( 
                nn.Parameter( 
                    torch.rand( 
                        (1, 
                        self.seq_len, 
                        self.output_size), 
                        requires_grad=True 
                        ).to(self.DEVICE)
                    ) 
                )

        self.dropout = nn.Dropout(.2)

        # Attention Block
        self.attention_blocks = []

        for _ in range( self.n_attention_blocks ):
            self.attention_blocks.append( self._generate_attention_block( self.img_embedding, self.output_size ) )
        
        # ENCODING OUTPUT
        self.logits_conv_w, self.logits_conv_baias =\
            self._generate_w_bias( self.output_size, self.compressed_features_size )

    def _generate_attention_block(self, input_size, output_size):        
        encoding_norm = nn.LayerNorm( self.output_size ).to(self.DEVICE)

        encoding_w, encoding_baias = self._generate_w_bias( output_size, output_size * 3 )
        merging_w, merging_baias = self._generate_w_bias( output_size, output_size )

        residual_norm = nn.LayerNorm( self.output_size ).to(self.DEVICE)

        residual_w1, residual_baias1 = self._generate_w_bias( output_size, output_size * 4 )
        residual_w2, residual_baias2 = self._generate_w_bias( output_size * 4, output_size )

        dropout = nn.Dropout( .2 )

        block = {
            'encoding_norm': encoding_norm,

            'encoding_w': encoding_w,
            'encoding_baias': encoding_baias,            
            'merging_w': merging_w,
            'merging_baias': merging_baias,

            'residual_norm': residual_norm,

            'residual_w1': residual_w1,
            'residual_baias1': residual_baias1,
            'residual_w2': residual_w2,
            'residual_baias2': residual_baias2,

            'dropout': dropout
        }

        return block

    def _generate_w_bias(self, input_size, output_size):
        
        # WEIGHTS
        w = kaiming_weight_init( 
            nn.Parameter( 
                torch.rand( 
                    (1, 
                    input_size, 
                    output_size), 
                    requires_grad=True 
                    ).to(self.DEVICE)
                ) 
            )
        
        # BAIAS
        b = torch.zeros( output_size, requires_grad=True ).to(self.DEVICE)

        return w, b

    def _custom_conv1(self, x, w, b):
        dims_x = x.shape
        dims_w = w.shape

        # RESHAPE FOR MLP FORWARD [ BATCH * SEQ, FEATURES ]
        xs = x.view( dims_x[0] * dims_x[1], dims_x[2] )

        # [ INPUT, OUTPUT ] => ITEM_FEATURES * 3  >FORWARD> ITEM_FEATURES * 3 * ATTENTION_HEADS
        wm = w.view( -1, dims_w[-1] )

        xs_ws = (xs @ wm) + b # MATRIZ MULTIPLICATION => input * weights + baias

        # RESHAPE FOR OUTPUT [ BATCH, SEQ, FEATURES ]
        x_out = xs_ws.view( dims_x[0], dims_x[1], dims_w[-1] )

        return x_out

    def _custom_conv2(self, x, w, b):
        
        dims_x = x.shape
        dims_w = w.shape

        xc = x.view( dims_x[0] * dims_x[1], dims_x[2], dims_x[3], dims_x[4] )

        xc = self.conv1( xc )
        xc = F.relu( xc )
        xc = self.pool1( xc )
                
        xc = self.conv2( xc )
        xc = F.relu( xc )
        xc = self.pool2( xc )

        xc = self.conv3( xc )
        xc = F.relu( xc )
        xc = self.pool3( xc )

        xc = xc.view( -1, self.flatten_size )        

        # RESHAPE FOR MLP FORWARD [ BATCH * SEQ, FEATURES ]
        # xs = x.view( dims_x[0] * dims_x[1], dims_x[2] )
        xs = xc

        # [ INPUT, OUTPUT ] => ITEM_FEATURES * 3  >FORWARD> ITEM_FEATURES * 3 * ATTENTION_HEADS
        wm = w.view( -1, dims_w[-1] )

        xs_ws = (xs @ wm) + b # MATRIZ MULTIPLICATION => input * weights + baias

        # RESHAPE FOR OUTPUT [ BATCH, SEQ, FEATURES ]
        x_out = xs_ws.view( dims_x[0], dims_x[1], dims_w[-1] )

        return x_out

    def forward(self, state, dropout=True):
        
        x = state
        dims_x = x.shape
        
        # FLATTEN
        x = x.view( dims_x[0], dims_x[1], dims_x[2] * dims_x[3] * dims_x[4] )
        
        # IMG Embedding
        x = self._custom_conv1( x, self.fc1_w, self.fc1_baias )
        x = F.relu( x )
        x = self._custom_conv1( x, self.fc2_w, self.fc2_baias )
        x = F.relu( x )
        x = self._custom_conv1( x, self.fc3_w, self.fc3_baias )
        x = F.relu( x )
        x = self._custom_conv1( x, self.fc4_w, self.fc4_baias )
        x = F.relu( x )
        x = self._custom_conv1( x, self.fc5_w, self.fc5_baias )
        x = torch.tanh( x )

        # GPT 2

        # [ BATCH, SEQ(N_ITEMS), FEATURES ]
        # x = x.view( dims_x[0], dims_x[1], self.flatten_size )

        # [ BATCH, SEQ(N_ITEMS), FEATURES X HEADS ]        
        x = self._custom_conv1( x, self.pre_enconding_w, self.pre_encoding_baias )

        # POSITIONAL SEQxFEATURES_CONV + WP
        x = x + self.positional_w

        if dropout:
            x = self.dropout( x )

        # ATTENTION
        for block in self.attention_blocks:
            # ATTENTION REQUIRES NORM        
            x_norm = block['encoding_norm']( x )

            c = self._custom_conv1( x_norm, block['encoding_w'], block['encoding_baias'] )

            Q, K, V = c.split( self.output_size, dim=2 )        

            Q = Q.view( -1, dims_x[1], self.attention_heads, self.output_size // self.attention_heads )
            K = K.view( -1, dims_x[1], self.attention_heads, self.output_size // self.attention_heads )
            V = V.view( -1, dims_x[1], self.attention_heads, self.output_size // self.attention_heads )

            Q = Q.transpose( 2, 1 )
            K = K.transpose( 2, 1 )
            V = V.transpose( 2, 1 )
        
            # ATTENTION CALC
            w = Q @ K.transpose( 3, 2 )
            w = w * torch.rsqrt( torch.tensor( V.shape[-1] ).float() )

            # MASK
            i = torch.arange( w.shape[2] ).view(-1, 1).to(self.DEVICE)
            j = torch.arange( w.shape[3] ).to(self.DEVICE)
            m = (i >= j - w.shape[2] + w.shape[3]).float()
            m = m.view(1, 1, m.shape[0], m.shape[1])

            # APPLYING MASK
            w = w * m - 1e10 * (1 - m)
            s = torch.softmax( w, dim=3 )

            # S = Q * K
            # S * V
            a = s @ V
            a = a.transpose( 1, 2 )
            a = a.reshape( -1, a.shape[1], a.shape[2] * a.shape[3] )

            # APPLYING ATTENTION TO THE INPUT
            x = x + a

            # END ATTENTION

            # POS ATTENTION NORM
            x = block['residual_norm']( x )

            # APPLYING RESIDUAL
            m = self._custom_conv1( x, block['residual_w1'], block['residual_baias1'] )
            m = F.gelu( m )
            m = self._custom_conv1( m, block['residual_w2'], block['residual_baias2'] )

            x = x + m

            if dropout:
                x = block['dropout']( x )

        # ENCODING OUTPUT
        encoded = self._custom_conv1( x, self.logits_conv_w, self.logits_conv_baias )

        # encoded = torch.tanh( encoded )
        encoded = ( encoded - encoded.mean() ) / encoded.std() + 1.0e-10

        return encoded

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class AttentionActionModel(nn.Module):

    def __init__(self, encoding_size, action_size, fc1_units=64, fc2_units=32, device='cpu'):
        super(AttentionActionModel, self).__init__()

        self.encoding_size = encoding_size
        self.action_size = action_size

        self.fc1 = layer_init( nn.Linear(encoding_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units + action_size, fc2_units) )
        # self.fc3 = layer_init( nn.Linear(fc2_units, 1) )
        self.fc3 = nn.Linear(fc2_units, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        
        x = self.fc1(state)
        x = F.relu( x )
        
        x = torch.cat( (x, action), dim=2 )

        x = self.fc2( x )
        x = F.relu( x )

        x = self.fc3( x ) # reward
        # x = torch.tanh( x )

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class DQNModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128, fc3_units=64, fc4_units=32):
        super(DQNModel, self).__init__() 

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        self.fc3 = layer_init( nn.Linear(fc2_units, fc3_units) )
        self.fc4 = layer_init( nn.Linear(fc3_units, fc4_units) )

        self.fc_action = layer_init( nn.Linear(fc4_units, action_size) )

    def forward(self, state):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = F.relu( self.fc4(x) )

        action_values = self.fc_action(x)

        return action_values

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class ActorModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=32):
        super(ActorModel, self).__init__() 

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.fc_action = layer_init( nn.Linear(fc2_units, action_size) )

    def forward(self, state, action=None):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )

        probs = F.softmax( self.fc_action(x), dim=1 )

        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        return action, log_prob, entropy

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class CriticModel(nn.Module):

    def __init__(self, state_size, fc1_units=64, fc2_units=32):
        super(CriticModel, self).__init__() 

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        
        self.fc_critic = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, state):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
        
        value = self.fc_critic(x)        

        return value

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

