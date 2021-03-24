import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from model.FC import FC
import dataset.Dataset as data

from utils.hparams import HParam
from utils.writer import MyWriter

def spec_to_wav(complex_ri, window, length):
    audio = torch.istft(input= complex_ri, n_fft=int(1024), hop_length=int(256), win_length=int(1024), window=window, center=True, normalized=False, onesided=True, length=length)
    return audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    ## Parameters
    device = hp.gpu
    torch.cuda.set_device(device)
    batch_size = hp.train.batch_size
    channels = hp.model.channels
    context = hp.model.context
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers
    best_loss = 10

    ## dirs

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    ## Logger
    writer = MyWriter(hp, log_dir)

    ## Data
    list_train= ['tr05_bus_simu','tr05_caf_simu','tr05_ped_simu','tr05_str_simu']
    list_test= ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']

    train_dataset = None
    test_dataset  = None

    if hp.feature == 'MFCC':
        train_dataset = data.dataset(hp.data.root+'/MFCC/',list_train,'*.pt',context=context,channels=channels)
        test_dataset  = data.dataset(hp.data.root+'/MFCC/',list_test,'*.pt',context=context,channels=channels)
    elif hp.feature == "LMPSC":
        train_dataset = data.dataset(hp.data.root+'/LMPSC/',list_train,'*.pt',context=context,channels=channels)
        test_dataset  = data.dataset(hp.data.root+'/LMPSC/',list_test,'*.pt',context=context,channels=channels)
    else :
        raise Exception('feature type is not available')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    ## Model
    model = None
    if hp.model.type == 'FC': 
        model = FC(hp).to(device)
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
            output = model(input)

            loss = criterion(output,target).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,"train_"+hp.loss.type)

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
                output = model(input)

                loss = criterion(output,target).to(device)

                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(test_loader), loss.item()))
                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            if hp.scheduler.type == 'Plateau' :
                scheduler.step(test_loss)

            #input_audio = wav_noisy[0].cpu().numpy()
            #target_audio= wav_clean[0].cpu().numpy()
            #audio_me_pe= audio_me_pe[0].cpu().numpy()

            writer.log_value(loss,step,"test_"+hp.loss.type)
            writer.log_MFCC(input[0],output[0],target[0],step)
            #input_audio,target_audio,audio_me_pe)
    

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

