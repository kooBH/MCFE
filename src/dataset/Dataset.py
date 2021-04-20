import os, glob
import torch
import numpy as np

class dataset(torch.utils.data.Dataset):
    def __init__(self, root,target,form, train=True,context=3,channels=2):
        self.root = root
        self.context = context
        self.context_length = context
        self.train = train
        self.channels = channels

        #self.data_list = [x for x in glob.glob(os.path.join(root,target,form),recursive=True) if not os.path.isdir(x)]
        if type(target) == str : 
            self.data_list = [x for x in glob.glob(os.path.join(root+'/noisy/', target, form), recursive=False) if not os.path.isdir(x)]
        elif type(target) == list : 
            self.data_list = []
            for i in target : 
                self.data_list = self.data_list + [x for x in glob.glob(os.path.join(root+'/noisy/', i, form), recursive=False) if not os.path.isdir(x)]
        else : 
            raise Exception('Unsupported type for target')

        # Extract id only. e.g. dt05_bus_simu/F01G1503
        for i in range(len(self.data_list)) : 
            tmp = self.data_list[i]
            tmp = tmp.split('/')
            self.data_list[i] = tmp[-2] + '/' + tmp[-1]
            self.data_list[i] = (self.data_list[i].split('.'))[0]

        self.data_list.sort(key=lambda x : x.split('/')[1])

    def __getitem__(self, index):
        path = self.data_list[index]

        # (Time, MFCC)
        pt_noisy = torch.load(self.root + '/noisy/'+path+'.pt')
        if self.channels >= 2 :
            pt_estim = torch.load(self.root + '/estim/'+path+'.pt')
        if self.channels >= 3 :
            pt_noise = torch.load(self.root + '/noise/'+path+'.pt')
        pt_clean = None
        if self.train : 
            pt_clean = torch.load(self.root + '/clean/'+path+'.pt')

        length = pt_noisy.shape[0]

        if self.train :
            center_idx = np.random.randint(low=0,high = length)
            # [ context ...  target ... context ]
            ## No need to pad
            if center_idx >= self.context_length and center_idx < length - self.context_length :
                pt_noisy = pt_noisy[center_idx-self.context_length:center_idx + self.context_length+1,:]
                if self.channels >= 2  :
                    pt_estim = pt_estim[center_idx-self.context_length:center_idx + self.context_length+1,:]
                if self.channels >= 3 :
                    pt_noise = pt_noise[center_idx-self.context_length:center_idx + self.context_length+1,:]
            ## padding on head
            elif center_idx < self.context_length :
                shortage = self.context_length - center_idx
                pad = torch.zeros(shortage,pt_noisy.shape[1])
                pt_noisy = torch.cat((pad,pt_noisy[center_idx -self.context_length+shortage:center_idx+self.context_length+1,:]),dim=0)
                if self.channels >= 2 :
                    pt_estim = torch.cat((pad,pt_estim[center_idx-self.context_length+shortage:center_idx+self.context_length+1]),dim=0)
                if self.channels >= 3 :
                    pt_noise= torch.cat((pad,pt_noisy[center_idx-self.context_length+shortage:center_idx+self.context_length+1,:]),dim=0)
            ## padding on tail
            elif center_idx >= length - self.context_length :
                shortage = center_idx - length + self.context_length + 1
                pad = torch.zeros(shortage,pt_noisy.shape[1])
                pt_noisy = torch.cat((pt_noisy[center_idx -self.context_length:length,:],pad),dim=0)
                if self.channels >= 2 :
                    pt_estim = torch.cat((pt_estim[center_idx-self.context_length:length,:],pad),dim=0)
                if self.channels >= 3 :
                    pt_noise= torch.cat((pt_noisy[center_idx-self.context_length:length,:],pad),dim=0)
            else :
                raise Exception("center_idx")
            
        input=None        
        if self.channels == 1 :
            input = pt_noisy
        elif self.channels == 2:
            input = torch.stack((pt_noisy,pt_estim),0)
        elif self.channels == 3 :
            input = torch.stack((pt_noisy,pt_estim,pt_noise),0)

        data = None
        if self.train : 
            data = {"input":input,"target":pt_clean}
        else :
            path = path.split('/')
            category = path[0]
            path = path[1]
            data = {"input":input,"name":path,'category':category}
        return data

    def __len__(self):
        return len(self.data_list)
