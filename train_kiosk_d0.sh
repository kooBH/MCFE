#!/bin/bash

#VERSION=TEST
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}


## 2021-05-22
#VERSION=MFCC_SKDAE_1
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}

## 2021-05-26
#VERSION=MFCC_SKDAE_2
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/nas/user/kbh/MCFE/chkpt/${VERSION}/bestmodel.pt -s 3890

## 2021-05-27
#VERSION=MFCC_SKDAE_1
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:0' 
#VERSION=MFCC_SKDAE_2
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:0' 
#VERSION=MFCC_SKDAE_3
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:0' 


#VERSION=MFCC_SKDAE_7
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:0' 

VERSION=MFCC_SKDAE_8
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/nas/user/kbh/MCFE/chkpt/${VERSION}/bestmodel.pt