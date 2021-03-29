#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tbsm_py="python ./profiling/tbsm_taobao_preprocess.py "

$tbsm_py  	--arch-sparse-feature-size=16 \
			--arch-embedding-size="987994-4162024-9439" \
			--arch-mlp-bot="1-16" \
			--arch-mlp-top="15-1" \
			--data-generation=dataset \
			--data-set=alibaba \
			--raw-data-file=./input/taobao_train.txt \
			--processed-data-file=./input/taobao_train_t20.npz \
			--loss-function=mse \
			--round-targets=False \
			--mini-batch-size=2 \
			--print-freq=1000 \
			--print-time \
			--threshold=0.000001
