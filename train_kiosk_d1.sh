#!/bin/bash

#VERSION=TEST
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}

## 2021-05-27
#VERSION=MFCC_SKDAE_5
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:1' --chkpt /home/nas/user/kbh/MCFE/chkpt/${VERSION}/bestmodel.pt -s 23710
#VERSION=MFCC_SKDAE_4
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:1' 
VERSION=MFCC_SKDAE_5
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:1' 
VERSION=MFCC_SKDAE_6
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}  -d 'cuda:1' 
