#!/bin/bash

# you can set the hparams by using --hparams=xxx
CUDA_LAUNCH_BLOCKING=1 python train.py -l logdir \
-o outdir --n_gpus=1 --hparams=speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.,training_list=/home/josh/vctk/processed/list.train,validation_list=/home/josh/vctk/processed/list.eval

