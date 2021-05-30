import torch
import argparse
import torchaudio
import os
import numpy as np
from kaldiio import WriteHelper

from tqdm import tqdm
from tensorboardX import SummaryWriter

from model.FC import FC
from model.SKDAE import SKDAE

import dataset.Dataset as data

from utils.hparams import HParam
from utils.writer import MyWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default='cuda:0')
    parser.add_argument('--outpath','-o',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    ## Parameters
    device = args.device
    #torch.cuda.set_device(device)
    batch_size = 1 
    context = hp.model.context
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers
    feature_size = hp.model.feature_size
    channels = 3

    ## Data
    list_dt05_simu = ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu']
    list_dt05_real = ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real']
    list_et05_simu = ['et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']
    list_et05_real = ['et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real']

    inference_dataset  = None

    if hp.feature == 'MFCC':
        inference_dataset  = [
            data.dataset(hp.data.inference,context=context,inference = list_dt05_simu),
            data.dataset(hp.data.inference,context=context,inference = list_et05_simu),
            data.dataset(hp.data.inference,context=context,inference = list_dt05_real),
            data.dataset(hp.data.inference,context=context,inference = list_et05_real)
        ]
    elif hp.feature == "LMPSC":
            inference_dataset  = [
            data.dataset(hp.data.root+'/LMPSC/inference/',list_dt05_simu,context=context,inference=True),
            data.dataset(hp.data.root+'/LMPSC/inference/',list_dt05_real,context=context,inference=True),
            data.dataset(hp.data.root+'/LMPSC/inference/',list_et05_simu,context=context,inference=True),
            data.dataset(hp.data.root+'/LMPSC/inference/',list_et05_real,context=context,inference=True)
        ]
    else :
        raise Exception('feature type is not available')

    inference_loader = [
        torch.utils.data.DataLoader(dataset=inference_dataset[0],batch_size=batch_size,shuffle=False,num_workers=num_workers),
        torch.utils.data.DataLoader(dataset=inference_dataset[1],batch_size=batch_size,shuffle=False,num_workers=num_workers),
        torch.utils.data.DataLoader(dataset=inference_dataset[2],batch_size=batch_size,shuffle=False,num_workers=num_workers),
        torch.utils.data.DataLoader(dataset=inference_dataset[3],batch_size=batch_size,shuffle=False,num_workers=num_workers),
    ]

    ## Model
    model = None
    if hp.model.type == 'FC': 
        model = FC(hp).to(device)
    elif hp.model.type == 'SKDAE':
        model = SKDAE(hp).to(device)
    else :
        raise Exception("hp.model.type unknown")

    print('NOTE::Loading pre-trained model : '+ args.chkpt)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))

    version = args.config.split('/')
    version = version[-1].split('.')[0]

    writer = None

    ##  params
    cnt = 0
    idx_category = 0
    list_category= [
        'dt05_simu',
        'dt05_real',
        'et05_simu',
        'et05_real'
    ]
    # Need to create ${nj_MFCC} of ark,scp pairs
    nj_MFCC = 8

    ## Inference
    model.eval()
    with torch.no_grad():
        for i in inference_loader :
            len_dataset = len(i)
            cur_category = list_category[idx_category]
            idx_category+=1
            max_item = int(len_dataset / nj_MFCC)
            if len_dataset % nj_MFCC != 0 :
                raise Exception('len_dataset must be devided by nj')
            ver = 1
            cnt = 0

            for j, data in enumerate(tqdm(i,desc=cur_category)):
                length = data['input'].shape[2]
                # [1, channels, length, feature_size]
                input = data['input']            
                output = None

    #            print('input shape 1 : ' + str(input.shape))

                ## run for a sample
                for k in range(length) :
                    ## padding on head
                    if k < context : 
                        shortage = context - k
                        pad = torch.zeros(1,channels,shortage,feature_size)
                        input_tmp = torch.cat((pad,input[:,:,k - context +shortage:k+context+1,:]),dim=2)
                        # padding on tail
                    elif k >= length - context :
                        shortage = k - length +context + 1
        #                   print('shortage : ' + str(shortage))
                        pad = torch.zeros(1,channels,shortage,feature_size)

        #                   print(input[:,:,k-context:length,:].shape)
        #                  print(pad.shape)

                        input_tmp = torch.cat((input[:,:,k-context:length,:],pad),dim=2)
        #                   print(input_tmp.shape)
                    else :
                        input_tmp = input[:,:,k-context:k+context+1,:]

                    input_tmp = input_tmp.to(device)
                    #print('input shape 2 : ' + str(input.shape))
                    output_frame = model(input_tmp)

    #                print(str(k)+': ' + str(input_tmp.shape))

                    if output == None :
                        # init
                        output = output_frame
    #                    print('output shape : ' + str(output_frame.shape))
                    else :
                        # concat
                        output  = torch.cat((output,output_frame),dim=0)
    #                   print('output shape : ' + str(output.shape))
                
                ## save as kaldi data type

                # name e.g. : F01_050C0101_PED_REAL
                name = data['name'][0]
                category = data['category'][0]
                category = category.split('_')
                real_simu = category[2].upper()
                category = category[0]+'_'+category[2]
                name = name + '_'+real_simu
                output = output.to(device)

                os.makedirs(args.outpath,exist_ok=True)
                #torch.save(output,args.outpath+'/'+name+'.pt')

                output = output.detach().cpu().numpy()

                ## kaldi MFCC with 13 feature_size
                if hp.feature == 'LMPSC':
                    # TODO
                    pass
                if feature_size > 13 :
                    # TODO
                    pass

                # ark,scp for each category
                # after 250 samples, new ark,scp
                if cnt == 0 :
                    filename= 'ark,scp:'+args.outpath+'/raw_mfcc_'+cur_category+'_'+version+'.'+str(ver)+'.ark,'+args.outpath+'/raw_mfcc_'+cur_category+'_'+version+'.'+str(ver)+'.scp'
                    writer= WriteHelper(filename,compression_method=2)

                writer(name,output)
                cnt += 1

                if cnt> max_item :
                    cnt = 0
                    ver +=1
