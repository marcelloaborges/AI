import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Encoder(nn.Module):
    def __init__(self, channels=8, compressed_features_size=32, fc1_units=256):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

        self.flatten_size = 32 * 10 * 9

        self.fc1_mu = nn.Linear( self.flatten_size, fc1_units )
        self.mu = nn.Linear( fc1_units, compressed_features_size )

        self.fc1_logvar = nn.Linear( self.flatten_size, fc1_units )
        self.logvar = nn.Linear( fc1_units, compressed_features_size )

    def forward(self, state):        
        # Encoder
        x = self.conv1( state )
        x = F.relu( x )
        x = self.conv2( x )
        x = F.relu( x )

        x = x.view( -1, self.flatten_size )        

        mu = self.fc1_mu( x )
        mu = F.relu( mu )
        mu = self.mu( mu )

        logvar = self.fc1_logvar( x )
        logvar = F.relu( logvar )
        logvar = self.logvar( logvar )
        
        return mu, logvar

    def load(self, checkpoint, device:'cpu'):     
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class Decoder(nn.Module):
    def __init__(self, channels=8, compressed_features_size=32, fc1_units=256):
        super(Decoder, self).__init__()        

        self.flatten_size = 32 * 10 * 9
        
        self.fc1 = nn.Linear( compressed_features_size, fc1_units)
        self.fc2 = nn.Linear( fc1_units, self.flatten_size)

        self.conv_t1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3,4), stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=(4,3), stride=3)

    def forward(self, out_features):
        x = self.fc1( out_features )
        x = F.relu( x )
        x = self.fc2( x )
        x = F.relu( x )

        x = x.view(-1, 32, 10, 9)
        
        x = self.conv_t1(x)
        x = F.relu(x)
        x = self.conv_t2(x)
        x = torch.sigmoid(x)

        return x
    
    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class DuelingDQN(nn.Module):

    def __init__(self, encoded_state_size, action_size, fc1_units=32):
        super(DuelingDQN, self).__init__()

        # FC
        self.fc1 = layer_init( nn.Linear(encoded_state_size, fc1_units) )

        self.actions = layer_init( nn.Linear(fc1_units, action_size) )
        self.value = layer_init( nn.Linear(fc1_units, 1) )

    def forward(self, state):        
        x = self.fc1(state)
        x = F.relu( x )
                        
        adv = self.actions( x )
        adv = adv - adv.mean()
                
        value = self.value(x)

        actions_values = adv + value

        return actions_values

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class DQN(nn.Module):

    def __init__(self, encoded_state_size, action_size, fc1_units=32):
        super(DQN, self).__init__()

        # FC
        self.fc1 = layer_init( nn.Linear(encoded_state_size, fc1_units) )

        self.actions = layer_init( nn.Linear(fc1_units, action_size) )

    def forward(self, state):        
        x = self.fc1(state)
        x = F.relu( x )
                        
        actions = self.actions( x )

        return actions

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)    

class Model(nn.Module):    

    def __init__(self, channels, compressed_features_size, fce_units, vae_samples, action_size, fc1_units):
        super(Model, self).__init__()

        self.VAE_SAMPLES = vae_samples

        self.encoder = Encoder(channels, compressed_features_size, fce_units)
        self.decoder = Decoder(channels, compressed_features_size, fce_units)

        self.dueling_dqn = DuelingDQN(compressed_features_size, action_size, fc1_units)

    def _reparameterize(self, mu, logvar, samples=4):
        samples_z = []

        for _ in range(samples):
            samples_z.append( self._z(mu, logvar) )    

        return samples_z

    def _z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)

        return z

    def forward(self, state, decode=True):
        mu_states, logvar_states = self.encoder(state)

        decoded_states = None 
        if decode:
            encoded_states = self._reparameterize( mu_states, logvar_states, self.VAE_SAMPLES )
            decoded_states = [ self.decoder( z ) for z in encoded_states ]            

        actions_values = self.dueling_dqn(mu_states)
        
        return actions_values, decoded_states, mu_states, logvar_states

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class Attention(nn.Module):

    def __init__(self, action_size, channels=8, fc1_units=64, fc2_units=32):
        super(Attention, self).__init__()
        
        self.channels = channels   
        self.filters_size = 16
        self.attention_heads = 4

        # Dimensionality
        self.conv_pre_attention = nn.Conv2d(in_channels=channels, out_channels=self.filters_size, kernel_size=3, padding=1 )        
        self.pool_pre_attention = nn.MaxPool2d( 2, 2 )

        # Attention
        self.WQs = []
        self.WKs = []
        self.WVs = []

        for _ in range(self.attention_heads):
            self.WQs.append(
                nn.Conv2d( 
                    in_channels=self.filters_size, 
                    out_channels=self.filters_size, 
                    kernel_size=3, padding=1 )
            )
            self.WKs.append(
                nn.Conv2d( 
                    in_channels=self.filters_size, 
                    out_channels=self.filters_size, 
                    kernel_size=3, padding=1 )
            )
            self.WVs.append(
                nn.Conv2d( 
                    in_channels=self.filters_size, 
                    out_channels=self.filters_size, 
                    kernel_size=3, padding=1 )
            )        

        self.Out_W = nn.Conv2d( in_channels=self.filters_size, out_channels=self.filters_size, kernel_size=3, padding=1 )

        # Encoding
        self.conv_pos_attention = nn.Conv2d(in_channels=self.filters_size, out_channels=self.filters_size, kernel_size=3, padding=1 )
        self.pool_pos_attention = nn.MaxPool2d( 2, 2 )

        self.flatten_size = self.filters_size * 21 * 80

        # DQN
        self.fc1 = layer_init( nn.Linear(self.flatten_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        
        self.actions = layer_init( nn.Linear(fc2_units, action_size) )        

    def forward(self, state):        
        x = state

        x = self.conv_pre_attention( x )
        x = F.relu( x )        
        x = self.pool_pre_attention( x )

        # Attention calc        
        Zs = []
        for i in range(self.attention_heads):
            z = self._attention_forward( x, self.WQs[i], self.WKs[i], self.WVs[i] )
            Zs.append( z )

        Zs = torch.cat( Zs, dim=3 )

        attention = self.Out_W( Zs )

        x = attention

        # Encoding        
        x = self.conv_pos_attention( x )
        x = F.relu( x )
        x = self.pool_pos_attention( x )

        x = x.view( -1, self.flatten_size )

        # Dueling DQN
        x = self.fc1( x )
        x = F.relu( x )
        x = self.fc2( x )
        x = F.relu( x )

        actions_values = self.actions( x )        

        return actions_values

    def _attention_forward(self, x, WQ, WK, WV):
        dims = x.shape

        # WQ, WK, WV
        Q = WQ( x ) 
        K = WK( x ) 
        V = WV( x )

        # Reshape for Scalar Mult
        Q = Q.view( dims[0], dims[1], dims[2] * dims[3] ) 
        K = K.view( dims[0], dims[1], dims[2] * dims[3] ) 
        V = V.view( dims[0], dims[1], dims[2] * dims[3] ) 

        Kt = K.transpose(1, 2)
        transpose_mul = torch.bmm( Q, Kt )
        regularization = transpose_mul / torch.sqrt( torch.tensor( self.channels ).float() )
        softmax_regularization = torch.softmax( regularization, dim=2 )
        Z = torch.bmm( softmax_regularization, V )        

        Z = Z.reshape( (dims[0], dims[1], dims[2], dims[3]) )

        return Z

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)