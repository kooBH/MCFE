#! /bin/bash  
KALDI=~/dnn2_kaldi/egs/chimd4_mod/s5_6ch_enhan_clean/mfcc/
VERSION=MFCC_SKDAE_5
python src/inference.py -c config/${VERSION}.yaml --chkpt /home/nas/user/kbh/MCFE/chkpt/${VERSION}/bestmodel.pt -d 'cuda:1' -o ${KALDI}/${VERSION}/