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
        # less layer -- 1
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
        # more layer -- 1
        elif self.version == 3:
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
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        # broader layer -- 1
        elif self.version == 4:
            node_size = 2048
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
        ## broader layer -- 2
        elif self.version == 5:
            node_size = 800
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        ## less layer -- 2
        elif self.version == 6 : 
            node_size = 400
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        ## Autoencoder
        elif self.version == 7 :
            node_size = 400
            bottleneck_size = 100
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,bottleneck_size),
                nn.BatchNorm1d(bottleneck_size),
                nn.Sigmoid(),
                nn.Linear(bottleneck_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        ## narrow model -- 2
        elif self.version == 8:
            node_size = 200
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.Sigmoid(),
                nn.Linear(node_size,self.fbank)
            )
        ## differenct activation -- 2
        elif self.version == 9:
            node_size = 400
            self.seq = nn.Sequential(
                nn.Linear(self.input_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.ReLU(),
                nn.Linear(node_size,node_size),
                nn.BatchNorm1d(node_size),
                nn.ReLU(),
                nn.Linear(node_size,self.fbank)
            )
        ## differenct activation -- 2
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
        print('x shape 2 : '+str(x.shape))
        return self.seq(x)