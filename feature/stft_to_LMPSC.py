# generic
import os, glob
# process
import numpy as np
import torch
import torchaudio
import librosa
import scipy
import scipy.io
import soundfile

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

fft_size = 400
shift_size = 160
mel_band = 23

input_root = '/home/data/kbh/MCFE/STFT/'
output_root= '/home/data/kbh/MCFE/LMPSC'+str(mel_band)+'/'

target_list = [x for x in glob.glob(os.path.join(input_root,'*','*','*.npy')) if not os.path.isdir(x)]

def  generate(idx):
    target_path = target_list[idx]
    # data shape [Time,Freq,Complex]
    target = np.load(target_path)
    target = target[:,:,0] + target[:,:,1]*1j
    # mel frequencies
    target =  librosa.feature.melspectrogram( S=target, n_fft=fft_size, hop_length=shift_size,win_length=None, window='hann', center=False, pad_mode='reflect', power=2.0, n_mels=mel_band)

    ## log mel
    target = 10*np.log10(np.power(target,2))

    # modify path
    # STFT/clean/dt05_str_simu/A.npy
    # => LMPSC/clean/dt05_str_simu/A.npy
    output_path = target_path.split('/')
    output_path = output_root + '/'+ output_path[-3]+'/'+output_path[-2]+'/'+output_path[-1].split('.')[0]+'.pt'

    target = torch.from_numpy(target)

    torch.save(target,output_path)

if __name__=='__main__' : 
    cpu_num = cpu_count()

    list_category = ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu','tr05_bus_simu','tr05_caf_simu','tr05_ped_simu','tr05_str_simu']
    list_real = ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real','et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real']
    list_dir = ['noisy','clean','estim']
    os.makedirs(output_root,exist_ok=True)
    for i in list_dir : 
        os.makedirs(output_root + i,exist_ok=True)
        for j in list_category :
            os.makedirs(output_root + i+'/'+j,exist_ok=True)
        for j in list_real :
            os.makedirs(output_root + i+'/'+j,exist_ok=True)

    arr = list(range(len(target_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(generate, arr), total=len(arr),ascii=True,desc='LMPSC'+str(mel_band)))






