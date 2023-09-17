#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

dlrm_fae_profiler_py="python ./dlrm_input_profiler.py "

$dlrm_fae_profiler_py  --arch-sparse-feature-size=16 \
						--arch-mlp-top="512-256-1" \
						--arch-mlp-bot="13-512-256-64-16" \
						--data-generation=dataset \
						--data-set=kaggle \
						--raw-data-file=./input/kaggle/train.txt \
						--processed-data-file=./input/kaggle/kaggleAdDisplayChallenge_processed.npz \
						--mini-batch-size=256 \
						--hot-emb-gpu-mem=268435456 \
						--ip-sampling-rate=5
