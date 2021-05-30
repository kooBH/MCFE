# Correlation Distance Skip Connection Denoising AutoEncoder
# Based on https://www.sciencedirect.com/science/article/pii/S0003682X19308175

import torch
import torch.nn as nn

import torch.nn.functional as F

class SKDAE(nn.Module):
    def __init__(self,hp):
        super().__init__()

        self.hp = hp
        self.context  = hp.model.context
        self.sz_feature = hp.model.feature_size
        self.sz_batch = hp.train.batch_size
        self.version  = hp.model.version
        self.channels = 3

        self.sz_input = self.channels*(2*self.context+1)*self.sz_feature
        self.sz_skip = self.channels*self.sz_feature

        if hp.model.activation == 'sigmoid' : 
            self.activation = nn.Sigmoid()
        elif hp.model.activation == 'tanh': 
            self.activation = nn.Tanh()
        else : 
            raise Exception('hp.model.activation unknown')

        # [B, C, T, F ]
        self.sz_layer=[]
        self.sz_layer.append(self.sz_input)
        self.sz_layer.append(512)
        self.sz_layer.append(256)
        self.sz_layer.append(128)

        self.encoder = []
        # input -> 512
        self.encoder.append(
            nn.Sequential(
            nn.Linear(self.sz_layer[0],self.sz_layer[1]),
            nn.BatchNorm1d(self.sz_layer[1]),
            self.activation
            )
        )
        # 512 -> 256
        self.encoder.append(
            nn.Sequential(
            nn.Linear(self.sz_layer[1],self.sz_layer[2]),
            nn.BatchNorm1d(self.sz_layer[2]),
            self.activation
            )
        )
        # 256 -> 128, skip-connection
        self.encoder.append(
            nn.Sequential(
            nn.Linear(self.sz_layer[2]+self.sz_skip,self.sz_layer[3]),
            nn.BatchNorm1d(self.sz_layer[3]),
            self.activation
            )
        )
        if hp.model.version == 2 :
            self.enhancer = nn.Sequential(
                    nn.LSTM(self.sz_layer[3],hidden_size = int(self.sz_layer[3]/2),num_layers=2,batch_first=True,bidirectional=True),
                    nn.BatchNorm1d(self.sz_layer[3]),
                    self.activation
            )

        else : 
            self.enhancer = nn.Sequential(
                    nn.Linear(self.sz_layer[3],self.sz_layer[3]),
                    nn.BatchNorm1d(self.sz_layer[3]),
                    self.activation
                )

        self.decoder = []
        # 128 -> 256
        self.decoder.append(
            nn.Sequential(
            nn.Linear(self.sz_layer[3],self.sz_layer[2]),
            nn.BatchNorm1d(self.sz_layer[2]),
            self.activation
            )
        )
        #  256 -> 512
        self.decoder.append(
            nn.Sequential(
            nn.Linear(self.sz_layer[2],self.sz_layer[1]),
            nn.BatchNorm1d(self.sz_layer[1]),
            self.activation
            )
        )
        # 512 -> feature_size
        self.decoder.append(
            nn.Sequential(
            nn.Linear(self.sz_layer[1]+self.sz_skip,self.sz_feature)
            )
        )
        
        # RuntimeError: Tensor for ‘out’ is on CPU, Tensor for argument #1 ‘self’ is on CPU, but expected them to be on GPU (while checking arguments for addmm)
        # https://discuss.pytorch.org/t/runtimeerror-tensor-for-out-is-on-cpu-tensor-for-argument-1-self-is-on-cpu-but-expected-them-to-be-on-gpu-while-checking-arguments-for-addmm/105453
        self.encoder = nn.ModuleList(self.encoder)
        self.enhancer = nn.ModuleList(self.enhancer)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        #print('x shape 1 : '+str(x.shape) + ' | ' + str(self.input_size))

        # [B, C, T, F ]
        x_hat = torch.reshape(x[:,:,self.context,:],(x.shape[0],self.sz_skip))
        x = torch.reshape(x,(x.shape[0],self.sz_layer[0]))

        # encoding
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x = self.encoder[2](torch.cat((x,x_hat),1))

        # enhancement
        if self.hp.model.version == 2 : 
            x = torch.unsqueeze(x,1)
            x = self.enhancer[0](x)
            x = x[0]
            x = torch.squeeze(x,1)
        else : 
            x = self.enhancer[0](x)

        # decoding
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        x = self.decoder[2](torch.cat((x,x_hat),1))

        #print('x shape 2 : '+str(x.shape))
        return x


