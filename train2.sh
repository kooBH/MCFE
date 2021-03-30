#!/bin/bash

#VERSION=TEST
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}
VERSION=MFCC_FC_6
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}
VERSION=MFCC_FC_7
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}
VERSION=MFCC_FC_8
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}
VERSION=MFCC_FC_9
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}
VERSION=MFCC_FC_10
python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION}
#python src/trainer.py -c config/${VERSION}.yaml -v ${VERSION} --chkpt /home/nas/user/kbh/MCFE/chkpt/${VERSION}/bestmodel.pt -s 14000
