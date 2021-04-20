import torch
import torch.nn as nn

import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self,hp):
        super(FC, self).__init__()

        self.channels = hp.model.channels
        self.context = hp.model.context
        self.fbank = hp.model.fbank
        self.batch_size = hp.train.batch_size
        self.version = hp.model.version

        # [B, C, T, F ]
        self.input_size = self.channels*(2*self.context+1)*self.fbank


        # DBN
        if self.version == 1:
            node_size = 1024
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        # DAE
        elif self.version == 2:
            node_size = 400
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        # DAE 2 
        elif self.version == 10:
            node_size = 400
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Tanh(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Tanh(),
                nn.Linear(node_size,self.fbank)
            )
        else :
            raise Exception("model version unknown")

    def forward(self, x):
        #print('x shape 1 : '+str(x.shape) + ' | ' + str(self.input_size))
        x = torch.reshape(x,(x.shape[0],self.input_size))
        #print('x shape 2 : '+str(x.shape))
        return self.seq(x)