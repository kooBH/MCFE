import os, glob
import torch
import numpy as np

class dataset(torch.utils.data.Dataset):
    def __init__(self, root,context=3,SNRs=None,inference=None):
        self.root = root
        self.context = context
        self.context_length = context
        self.inference = inference
        self.SNRs = SNRs

        if SNRs : 
            if type(SNRs) == list : 
                self.data_list = []
                for i in SNRs : 
                    self.data_list = self.data_list + [x for x in glob.glob(os.path.join(root,i,'noisy', '*.pt'), recursive=False)]
            else : 
                raise Exception('Unsupported type for target(s)')
        elif inference : 
            if type(inference) == list : 
                self.data_list = []
                for i in inference : 
                    self.data_list = self.data_list + [x for x in glob.glob(os.path.join(root,'noisy',i,'*pt'))]
            else :
                raise Exception('Unsupported type for target(s)')

        else : 
            raise Exception('SNRs=None, infrence=None : No targets')

    def __getitem__(self, index):
        path = self.data_list[index]
        file_name = path.split('/')[-1]
        file_id = file_name.split('.')[0]


        # (Time, MFCC)
        if not self.inference : 
            SNR = path.split('/')[-3]
            pt_noisy = torch.load(os.path.join(self.root,SNR,'noisy',file_name))
            pt_estim = torch.load(os.path.join(self.root,SNR,'estim',file_name))
            pt_noise = torch.load(os.path.join(self.root,SNR,'noise',file_name))
            pt_clean = torch.load(os.path.join(self.root,SNR,'clean',file_name))
        else :
            category = path.split('/')[-2]
            pt_noisy = torch.load(os.path.join(self.root,'noisy',category,file_name))
            pt_estim = torch.load(os.path.join(self.root,'estim',category,file_name))
            pt_noise = torch.load(os.path.join(self.root,'noise',category,file_name))

        length = pt_estim.shape[0]

        center_idx = np.random.randint(low=0,high = length)
        # [ context ...  target ... context ]
        ## No need to pad
        if center_idx >= self.context_length and center_idx < length - self.context_length :
            pt_estim = pt_estim[center_idx-self.context_length:center_idx + self.context_length+1,:]
            pt_noisy = pt_noisy[center_idx-self.context_length:center_idx + self.context_length+1,:]
            pt_noise = pt_noise[center_idx-self.context_length:center_idx + self.context_length+1,:]
        ## padding on head
        elif center_idx < self.context_length :
            shortage = self.context_length - center_idx
            pad = torch.zeros(shortage,pt_noisy.shape[1])
            pt_estim = torch.cat((pad,pt_estim[center_idx -self.context_length+shortage:center_idx+self.context_length+1,:]),dim=0)
            pt_noisy = torch.cat((pad,pt_noisy[center_idx -self.context_length+shortage:center_idx+self.context_length+1,:]),dim=0)
            pt_noise= torch.cat((pad,pt_noise[center_idx-self.context_length+shortage:center_idx+self.context_length+1,:]),dim=0)
        ## padding on tail
        elif center_idx >= length - self.context_length :
            shortage = center_idx - length + self.context_length + 1
            pad = torch.zeros(shortage,pt_noisy.shape[1])
            pt_estim = torch.cat((pt_estim[center_idx -self.context_length:length,:],pad),dim=0)
            pt_noisy = torch.cat((pt_noisy[center_idx -self.context_length:length,:],pad),dim=0)
            pt_noise= torch.cat((pt_noise[center_idx-self.context_length:length,:],pad),dim=0)
        else :
            raise Exception("center_idx")
        
        input = torch.stack((pt_noisy,pt_estim,pt_noise),0)
        input = input.float()

        if not self.inference : 
            pt_clean = pt_clean[center_idx,:]
            data = {"input":input,"target":pt_clean}
        else :
            data = {"input":input,"name":file_id,"category":category}
        return data

    def __len__(self):
        return len(self.data_list)

    def load_sample(self):
        path = self.data_list[0]
        file_name = path.split('/')[-1]
        SNR = path.split('/')[-3]


        # (Time, MFCC)
        pt_noisy = torch.load(os.path.join(self.root,'SNR0','noisy',file_name))
        pt_estim = torch.load(os.path.join(self.root,'SNR0','estim',file_name))
        pt_noise = torch.load(os.path.join(self.root,'SNR0','noise',file_name))
        pt_clean = torch.load(os.path.join(self.root,'SNR0','clean',file_name))

        length = pt_clean.shape[0]

        pt_noisy = pt_noisy[:length,:]

        input = torch.stack((pt_noisy,pt_estim,pt_noise),0)

        input = input[:,:300,:]
        pt_clean = pt_clean[:300,:]


        return input, pt_clean

