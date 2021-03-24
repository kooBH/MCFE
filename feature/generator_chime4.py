# generic
import os, glob
# process
import numpy as np
import torch
import torchaudio
import librosa
import scipy
import scipy.io
import soundfile as sf
# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

fft_size = 400
shift_size = 160
window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)

noisy_root = '/home/data/kbh/isolated/'
estimated_root = '/home/data/kbh/CHiME4_CGMM_RLS/trial_04/'
clean_root = '/home/data/kbh/isolated_ext/'
real_root = '/home/data/kbh/isolated_1ch_track/'

output_root = '/home/data/kbh/MCFE/'

clean_list = [x for x in glob.glob(os.path.join(clean_root, '*', '*CH1.Clean.wav')) if not os.path.isdir(x)]
real_list = [x for x in glob.glob(os.path.join(real_root, '*', '*.wav') )]

## for Syncronization ##
def cross_correlation_using_fft(x, y):
    f1 = np.fft.fft(x)
    f2 = np.fft.fft(np.flipud(y))
    cc = np.real(np.fft.ifft(f1 * f2))
    return np.fft.fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x 
# shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

def generate(idx):
    clean_path = clean_list[idx]
    dir = clean_path.split('/')[-2]
    item = clean_path.split('/')[-1]
    item = item.split('.')
    item = item[0]
    
    # parsing
    if dir.startswith('tr') :
        tmp = item.split('_')
        tmp = tmp[0]+'_'+tmp[1]
        estimated_path = estimated_root + dir +'/'+ tmp + '.wav'
    else : 
        estimated_path = estimated_root + dir +'/'+ item + '.wav'
    noisy_path = noisy_root + dir + '/' + item + '.CH1.wav'
    
    noisy, sr = librosa.load(noisy_path,sr=16000)
    estim, sr = librosa.load(estimated_path,sr=16000)
    clean,sr = librosa.load(clean_path,sr=16000)
    
    
    # Normalization
    max_val = np.max(np.abs(noisy))
    noisy = noisy / max_val
    max_val = np.max(np.abs(estim))
    estim = estim / max_val
    max_val = np.max(np.abs(clean))
    clean = clean /max_val

    # Syncronization Due to initial overlap delay
    diff = compute_shift(estim,noisy)
    estim = estim[-diff:]
    noisy = noisy[:len(estim)]
    clean = clean[:len(estim)]
    
    # Spectra
    spec_noisy = librosa.stft(noisy,window='hann', n_fft=fft_size, hop_length=shift_size , win_length=None ,center=False)
    spec_estim = librosa.stft(estim,window='hann', n_fft=fft_size, hop_length=shift_size, win_length=None ,center=False)
    spec_clean = librosa.stft(clean,window='hann', n_fft=fft_size, hop_length=shift_size , win_length=None ,center=False)
    

    # Sync Masked 
    spec_noisy = np.complex64(spec_noisy)
    spec_estim = np.complex64(spec_estim)
    spec_clean = np.complex64(spec_clean)

    spec_noisy = np.concatenate((np.expand_dims(spec_noisy.real,-1),np.expand_dims(spec_noisy.imag,-1)),2)
    spec_estim = np.concatenate((np.expand_dims(spec_estim.real,-1),np.expand_dims(spec_estim.imag,-1)),2)
    spec_clean = np.concatenate((np.expand_dims(spec_clean.real,-1),np.expand_dims(spec_clean.imag,-1)),2)
    
    # save
    np.save(output_root+'/STFT/noisy/'+dir+'/'+item+'.npy',(spec_noisy))
    np.save(output_root+'/STFT/clean/'+dir+'/'+item+'.npy',(spec_clean))
    np.save(output_root+'/STFT/estim/'+dir+'/'+item+'.npy',(spec_estim))

def generateReal(idx):
    real_path = real_list[idx]
    dir = real_path.split('/')[-2]
    item = real_path.split('/')[-1]
    item = item.split('.')
    item = item[0]
    
    estimated_path = estimated_root + dir +'/'+ item + '.wav'
    noisy_path = noisy_root + dir + '/' + item + '.CH1.wav'
    
    noisy, sr = librosa.load(noisy_path,sr=16000)
    estim, sr = librosa.load(estimated_path,sr=16000)
    
    
    # Normalization
    max_val = np.max(np.abs(noisy))
    noisy = noisy / max_val
    max_val = np.max(np.abs(estim))
    estim = estim / max_val

    # Syncronization
    diff = compute_shift(estim,noisy)
    estim = estim[-diff:]
    noisy = noisy[:len(estim)]
    
    # Spectra
    spec_noisy = librosa.stft(noisy,window='hann', n_fft=fft_size, hop_length=shift_size , win_length=None ,center=False)
    spec_estim = librosa.stft(estim,window='hann', n_fft=fft_size, hop_length=shift_size , win_length=None ,center=False)
    
    
    # Sync Masked 
    spec_noisy = np.complex64(spec_noisy)
    spec_estim = np.complex64(spec_estim)

    spec_noisy = np.concatenate((np.expand_dims(spec_noisy.real,-1),np.expand_dims(spec_noisy.imag,-1)),2)
    spec_estim = np.concatenate((np.expand_dims(spec_estim.real,-1),np.expand_dims(spec_estim.imag,-1)),2)
    
    # save
    np.save(output_root+'/STFT/noisy/'+dir+'/'+item+'.npy',(spec_noisy))
    np.save(output_root+'/STFT/estim/'+dir+'/'+item+'.npy',(spec_estim))


if __name__=='__main__' : 
    
    list_category = ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu','tr05_bus_simu','tr05_caf_simu','tr05_ped_simu','tr05_str_simu']

    list_real = ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real','et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real']

    list_dir = ['noisy','clean','estim']

# Directory managing for STFT
    for i in list_dir : 
        for j in list_category :
                os.makedirs(os.path.join(output_root,'STFT' ,i,j),exist_ok=True)
        for j in list_real :
                os.makedirs(os.path.join(output_root,'STFT' ,i,j),exist_ok=True)
 
    cpu_num = cpu_count()
    # save 8 threads for others
    cpu_num = cpu_num - 8
    
    arr = list(range(len(clean_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(generate, arr), total=len(arr),ascii=True,desc='Processing_simu'))

    arr = list(range(len(real_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(generateReal, arr), total=len(arr),ascii=True,desc='Processing_real'))


