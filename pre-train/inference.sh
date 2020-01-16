#!/bin/bash

# you can set the hparams by using --hparams=xxx
CUDA_LAUNCH_BLOCKING=1 python inference.py --hparams=test_list=/home/josh/vctk/processed/list.test

