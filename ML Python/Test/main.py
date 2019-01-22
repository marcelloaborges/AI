import numpy as np

import torch

from model import Model

FEED_SIZE = 10
OUTPUT_SIZE = 2

model = Model(FEED_SIZE, OUTPUT_SIZE)

feed1 = np.random.rand( 1, FEED_SIZE ) 
feed2 = np.random.rand( 1, FEED_SIZE ) 
feed = np.stack( (feed1, feed2), axis=1 )

print( feed1 )
print( feed2 )
print( feed )


feed = torch.from_numpy( feed ).float()
out = model(feed).detach().numpy()

print( out )