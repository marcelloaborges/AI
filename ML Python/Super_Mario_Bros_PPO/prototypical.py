import numpy as np

import torch
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Prototypical:

    def __init__(
        self, 
        device,
        batch_size_prot,
        encoder_model,
        encoder_optimizer
        ):

        self.DEVICE = device

        self.BATCH_SIZE = batch_size_prot

        self.encoder_model = encoder_model     

        self.encoder_optimizer = encoder_optimizer
        self.scaler = GradScaler()   

        self.memory = {}
        for i in range(int(3158/10)):
            self.memory[i] = {
                'support': np.zeros([1, 4, 84, 84]),
                'query': np.zeros([1, 4, 84, 84])
            }

    def encode(self, state):
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.encoder_model.eval()

        with torch.no_grad():                            
            encoded = self.encoder_model(state)

        self.encoder_model.train()

        encoded = encoded.cpu().data.numpy()

        return encoded

    def step(self, state, x_pos):
        seg = int(x_pos / 10)
        if x_pos in self.memory:
            self.memory[seg]['query'] = state
        else:
            self.memory[seg] = {
                'support': state,
                'query': state
            }

    def learn(self):
        
        with autocast():
            x_support = []
            x_query = []
            for key, value in self.memory.items():
                x_support.append( value['support'] )    
                x_query.append( value['query'] )

            xs = torch.tensor( x_support ).float().to(self.DEVICE)
            xq   = torch.tensor( x_query   ).long().to(self.DEVICE)

            loss = 0.1
            acc = 0.1        
            
            n_class = xs.size(0)
            assert xq.size(0) == n_class
            n_support = xs.size(1)
            n_query = xq.size(1)

            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(self.DEVICE)

            x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                        xq.view(n_class * n_query, *xq.size()[2:])], 0)

            z = self.encoder_model(x)
            z_dim = z.size(-1)

            z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
            zq = z[n_class*n_support:]

            dists = self._euclidean_dist(zq, z_proto)

            log_p_y = F.log_softmax(-dists, dim=0).view(n_class, n_query, -1)

            loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

            _, y_hat = log_p_y.max(2)
            acc = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            

            self.encoder_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.encoder_optimizer)
            self.scaler.update()

            loss = loss.cpu().data.item()
            acc = acc.cpu().data.item()


        return loss, acc

    def _euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def _euclidean_dist2(self, x, y):        
        return torch.pow(x - y, 2).sum(1)