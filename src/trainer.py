import torch
import argparse
import torchaudio
import os
import numpy as np

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
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default='cuda:0')
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    ## Parameters
    device = args.device
    torch.cuda.set_device(device)
    batch_size = hp.train.batch_size
    context = hp.model.context
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers
    best_loss = 100000
    channels = 3
    feature_size = hp.model.feature_size

    ## dirs

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    ## Logger
    writer = MyWriter(hp, log_dir)

    ## Data
    SNRs= hp.data.SNR
    train_dataset = None
    test_dataset  = None

    if hp.feature == 'MFCC':
        train_dataset = data.dataset(hp.data.root+'/MFCC/train/',SNRs,context=context)
        test_dataset= data.dataset(hp.data.root+'/MFCC/test/',SNRs,context=context)
    else :
        raise Exception('feature type is not available')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    ## Model
    model = None
    if hp.model.type == 'FC': 
        model = FC(hp).to(device)
    elif hp.model.type == 'SKDAE':
        model = SKDAE(hp).to(device)
    else :
        raise Exception("Model == None")

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    ## Criterion
    criterion = None
    if hp.loss.type == 'MSE' :
        criterion = torch.nn.MSELoss()
    elif hp.loss.type == 'L1':
        criterion = torch.nn.L1Loss()
    else:
        raise Exception("Loss == None")

    ## Optimizer
    optimizer = None
    if hp.optim.type == 'Adam' :
       optimizer = torch.optim.Adam(model.parameters(), lr=hp.optim.adam)
    else :
        raise Exception("optimizer == None")

    ## Scheduler
    scheduler = None
    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
                )
    else :
        pass # No scheduler

    ### TRAIN ####
    step = args.step
    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1
            input = batch_data['input'].to(device)
            target = batch_data['target'].to(device)
            #mask = model(input)
            #output = input[:,target_idx,context,:]*mask
            output = model(input)

            loss = criterion(output,target).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,"train_"+hp.loss.type)

        print('TRAIN::Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs, step,loss.item()))

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')

        if hp.scheduler.type =='oneCycle':
            scheduler.step()
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (batch_data) in enumerate(test_loader):
                input = batch_data['input'].to(device)
                target = batch_data['target'].to(device)
                #mask = model(input)
                #output = input[:,target_idx,context,:]*mask
                output = model(input)

                loss = criterion(output,target).to(device)

                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            if hp.scheduler.type == 'Plateau' :
                scheduler.step(test_loss)

            print('TEST::Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

            writer.log_value(loss,step,"test_"+hp.loss.type)

            ## MFCC plot for a sample
            sample_input,sample_clean = test_dataset.load_sample()
            sample_input = torch.unsqueeze(sample_input,0)
            sample_length = sample_input.shape[2]
            sample_output = None
            sample_mask = None
            ## run for a sample
            for k in range(sample_length) :
                if k < context : 
                    shortage = context - k
                    pad = torch.zeros(1,channels,shortage,feature_size)
                    sample_input_tmp = torch.cat((pad,sample_input[:,:,k - context +shortage:k+context+1,:]),dim=2)
                elif k >= sample_length - context :
                    shortage = k - sample_length +context + 1
                    pad = torch.zeros(1,channels,shortage,feature_size)
                    sample_input_tmp = torch.cat((sample_input[:,:,k-context:sample_length,:],pad),dim=2)
                else :
                    sample_input_tmp = sample_input[:,:,k-context:k+context+1,:]

                sample_input_tmp = sample_input_tmp.to(device)
                # inference
                #mask = model(sample_input_tmp)
                #sample_output_frame  = sample_input_tmp[0,target_idx,context,:] * mask
                sample_output_frame = model(sample_input_tmp)

                output = model(input)
                # init
                if sample_output == None :
                    sample_output = sample_output_frame
                    #sample_mask = mask
                # concat
                else :
                    sample_output  = torch.cat((sample_output,sample_output_frame),dim=0)
                    #sample_mask = torch.cat((sample_mask,mask),dim=0)

            writer.log_MFCC(sample_input[0][0],'noisy',step)
            writer.log_MFCC(sample_input[0][1],'estim',step)
            #writer.log_Mask(sample_mask,'mask',step)
            writer.log_MFCC(sample_output,'enhanced',step)
            writer.log_MFCC(sample_clean,'clean',step)

            torch.save(model.state_dict(), str(modelsave_path)+'/s'+str(step)+'_l'+str(test_loss)+'.pt')

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

