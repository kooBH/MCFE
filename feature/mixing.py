import os,sys
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.append("../src")

from ptUtils.audio import Audio
from ptUtils.hparams import HParam
import ptUtils.mixer as mx
import soundfile as sf

noise_dir = ''
clean_dir = ''

def mix(hp, audio, target, noise, num,train):
    srate = hp.audio.sample_rate
    clean_dir = os.path.join(hp.data.vfws_dir,'clean', 'train' if train else 'test')
    mixed_dir = os.path.join(hp.data.vfws_dir,'mixed', 'train' if train else 'test')
    # w1 : target input
    # w2 : background noise
    w1, _ = librosa.load(target, sr=srate)
    w2, _ = librosa.load(noise, sr=srate)

    # trim 
    w1, _ = librosa.effects.trim(w1, top_db=20)
    

    # For batched trainig fix L value
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1 = w1[:L]

    mixed = mx.mix_noise(w1,w2,snr=0)
    
    ### Is it necessary ? 
    # norm = np.max(np.abs(mixed)) * 1.1
    # w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    ### save wav files
    target_wav_path = formatter(clean_dir, hp.form.target.wav, num)
    mixed_wav_path = formatter(mixed_dir, hp.form.mixed.wav, num)
    
    # librosa.output is deprecated in librosa version0.8.0
    sf.write(target_wav_path, w1, srate)
    sf.write(mixed_wav_path, mixed, srate)

    ### save magnitude spectrograms
    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)
    
    target_mag_path = formatter(clean_dir, hp.form.target.mag, num)
    mixed_mag_path = formatter(mixed_dir, hp.form.mixed.mag, num)
    
    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--config', type=str, required=True,
    parser.add_argument('-i', '--input', type=str, required=True,help="input dir")
    parser.add_argument('-i', '--output', type=str, required=True,help="output dir")
    args = parser.parse_args()

    #hp = HParam(args.config)
    #hp = HParam('../config/mixing.yaml')
    
    vfws_dir = hp.data.vfws_dir
    
    clean_dir = os.path.join(vfws_dir,'clean')
    mixed_dir = os.path.join(vfws_dir,'mixed')
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(mixed_dir, exist_ok=True)
    
    os.makedirs(os.path.join(clean_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(mixed_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(mixed_dir, 'test'), exist_ok=True)

    cpu_num = cpu_count()
    
    norm_dir = hp.data.norm_dir
    
    train_spk = [x for x in glob.glob(os.path.join(norm_dir, 'train','**'), recursive=True) if not os.path.isdir(x)]
    train_spk = [x for x in train_spk if len(x) >= 2]
    
    test_spk = [x for x in glob.glob(os.path.join(norm_dir, 'dev','**'), recursive=True) if not os.path.isdir(x)]
    test_spk = [x for x in test_spk if len(x) >= 2]

    audio = Audio(hp)

    noise_files = [x for x in glob.glob(os.path.join(hp.data.noise_dir,'**'), recursive=True) if not os.path.isdir(x)]

    def train_wrapper(num):
        target = train_spk[num]
        noise_idx = np.random.randint(len(noise_files))
        noise = noise_files[noise_idx]
        mix(hp, audio, target, noise, num, train=True)

    def test_wrapper(num):
        target = test_spk[num]
        noise_idx = np.random.randint(len(noise_files))
        noise = noise_files[noise_idx]
        mix(hp, audio, target, noise, num, train=False)

    arr = list(range(len(train_spk)))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

    arr = list(range(len(test_spk)))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))
