# Correlation Distance Skip Connection Denoising AutoEncoder
# Based on https://www.sciencedirect.com/science/article/pii/S0003682X19308175

import torch
import torch.nn as nn

import torch.nn.functional as F

# y : output
# x : target
# z : latent
def CDloss(y,x,z,alpha=0.01,beta=0.01):
    def CorrelationDistnace(a,b):
        a_kl = torch. 
        return

    loss_linear =  (1-Correlation(z,x)) + (1-CorrelationDistnace(y,x))
    loss_nonlinear (1-CorrelationDistnace(z,x))**2 + (1-CorrelationDistnace(y,x)**2)
    loss_mse = torch.nn.MSEloss(y,x)

    return loss_MSE + beta*loss_nonlinear + alpha*loss_linear

class SKDAE(nn.Module):
    def __init__(self,hp):
        super(DAE, self).__init__()

        self.channels = hp.model.channels
        self.context = hp.model.context
        self.fbank = hp.model.fbank
        self.batch_size = hp.train.batch_size
        self.version = hp.model.version

        # [B, C, T, F ]
        self.input_size = self.channels*(2*self.context+1)*self.fbank

    def forward(self, x):
        #print('x shape 1 : '+str(x.shape) + ' | ' + str(self.input_size))
        x = torch.reshape(x,(x.shape[0],self.input_size))

        #x_hat = 

        nn.Linear(self.input_size,512)
        nn.BatchNorm1d(512)

        nn.Linear(512,256)
        nn.BatchNorm1d(256)

        #print('x shape 2 : '+str(x.shape))
        return self.seq(x)


