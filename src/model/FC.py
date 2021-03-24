import torch
import torch.nn as nn

from utils.hparams import HParam
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
                nn.Linear(node_size,self.fbank),
            )
        elif self.version == 2:
            node_size = 400
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank),
            )
        else :
            raise Exception("model version unknown")

    def forward(self, x):
        x = torch.reshape(x,(x.shape[0],self.input_size))
        return self.seq(x)