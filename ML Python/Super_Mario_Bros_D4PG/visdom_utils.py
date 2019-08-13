import torch
import visdom

class VisdomI:

    def __init__(self, config='default'):
        
        self.vis = visdom.Visdom()

        self.options = None

        if config == 'default':            
            options=dict(
                # fillarea=True,
                # showlegend=False,
                width=1920,
                # height=800,
                # xlabel='Time',
                # ylabel='Volume',
                # ytype='log',
                title='loss',
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
            )        
    
    def track(self, step, value, win):
        '''step, value, are Tensor'''

        # win_plot = self.vis.line( opts = self.options )

        self.vis.line( X=torch.Tensor( [step] ), Y=torch.Tensor( [value] ), win=win, update='append' )