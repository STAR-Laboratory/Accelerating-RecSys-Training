#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tbsm_fae_profiler_py="python ./profiling/tbsm_fae_profiler.py "

$tbsm_fae_profiler_py  --arch-sparse-feature-size=16 \
						--arch-mlp-top="60-1" \
						--arch-mlp-bot="1-16" \
						--data-generation=dataset \
						--data-set=alibaba \
						--raw-data-file=./input/taobao_train.txt \
						--processed-data-file=./output/taobao_train_t20.npz \
						--mini-batch-size=256 \
						--hot-emb-gpu-mem=268435456 \
						--ip-sampling-rate=20
