# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
from itertools import repeat
# import bisect
# import shutil
import time
import json
import os
import copy
import math
import subprocess
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pool, Manager, Queue, Lock, current_process
from multiprocessing import shared_memory
import pandas as pd
import copy

# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=DeprecationWarning)

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

def dash_separated_ints(value):
	vals = value.split('-')
	for val in vals:
		try:
			int(val)
		except ValueError:
			raise argparse.ArgumentTypeError(
				"%s is not a valid dash separated list of ints" % value)

	return value


def dash_separated_floats(value):
	vals = value.split('-')
	for val in vals:
		try:
			float(val)
		except ValueError:
			raise argparse.ArgumentTypeError(
				"%s is not a valid dash separated list of floats" % value)

	return value

if __name__ == "__main__":
	### import packages ###
	import sys
	import argparse

	### parse arguments ###
	parser = argparse.ArgumentParser(
		description="Train Deep Learning Recommendation Model (DLRM)"
	)
	# model related parameters
	parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
	parser.add_argument(
		"--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
	# j will be replaced with the table number
	parser.add_argument(
		"--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
	parser.add_argument(
		"--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
	parser.add_argument(
		"--arch-interaction-op", type=str, choices=['dot', 'cat'], default="dot")
	parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
	# data
	parser.add_argument(
		"--data-generation", type=str, default="random"
	)  # synthetic or dataset
	parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
	parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
	parser.add_argument("--raw-data-file", type=str, default="")
	parser.add_argument("--processed-data-file", type=str, default="")
	parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
	parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
	parser.add_argument("--max-ind-range", type=int, default=-1)
	parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
	parser.add_argument("--num-indices-per-lookup", type=int, default=10)
	parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--memory-map", action="store_true", default=False)
	parser.add_argument("--dataset-multiprocessing", action="store_true", default=False,
						help="The Kaggle dataset can be multiprocessed in an environment \
						with more than 7 CPU cores and more than 20 GB of memory. \n \
						The Terabyte dataset can be multiprocessed in an environment \
						with more than 24 CPU cores and at least 1 TB of memory.")
	# mlperf logging (disables other output and stops early)
	parser.add_argument("--mlperf-logging", action="store_true", default=False)
	# stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
	parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
	# stop at target AUC Terabyte (no subsampling) 0.8025
	parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
	parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
	parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
	# training
	parser.add_argument("--mini-batch-size", type=int, default=1)
	# debugging and profiling
	parser.add_argument("--print-freq", type=int, default=1)
	parser.add_argument("--test-freq", type=int, default=-1)
	parser.add_argument("--test-mini-batch-size", type=int, default=1)
	parser.add_argument("--test-num-workers", type=int, default=0)
	parser.add_argument("--print-time", action="store_true", default=False)
	parser.add_argument("--debug-mode", action="store_true", default=False)
	parser.add_argument("--enable-profiling", action="store_true", default=False)
	parser.add_argument("--plot-compute-graph", action="store_true", default=False)
	# Input Profiling
	# Percentage Threshold
	parser.add_argument("--hot-emb-gpu-mem", type=int, default=268435456, help="GPU memory for hot embeddings") #536870912 (512MB), 268435456 (256MB), 134217728 (128MB)
	parser.add_argument("--ip-sampling-rate", type=int, default=5, help="Input sampling rate (in %)")
	args = parser.parse_args()

	### main loop ###

	def time_wrap():
		return time.time()

	# Using CPU only for input profiling
	device = torch.device("cpu")
	print("Using CPU...")

	# Input Profiling for datasets only
	train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
	
	ln_emb = train_data.counts
	
	# enforce maximum limit on number of vectors per embedding
	if args.max_ind_range > 0:
		ln_emb = np.array(list(map(
			lambda x: x if x < args.max_ind_range else args.max_ind_range,
			ln_emb
		)))

	train = []
	for i, train_tuple in enumerate(train_data):
		X, lS_i, T = train_tuple
		lS_i = np.array(lS_i).astype(np.float32)
		train.append([X, lS_i, T])
	train = np.array(train, dtype = object)
	train = train.tolist()
	
	X_bytes = train[0][0].nbytes
	lS_i_bytes = train[0][1].nbytes
	T_bytes = train[0][2].nbytes

	input_bytes = X_bytes + lS_i_bytes + T_bytes

	# Shared Memories for Multiprocessing based final input classification
	shm_train_hot = shared_memory.SharedMemory(create = True, size = input_bytes * len(train))
	train_hot_array = np.ndarray(len(train), dtype = object, buffer = shm_train_hot.buf)

	shm_train_normal = shared_memory.SharedMemory(create = True, size = input_bytes * len(train))
	train_normal_array = np.ndarray(len(train), dtype = object, buffer = shm_train_normal.buf)

	def single_process_ip_classification(train_data, hot_emb_dict, train_hot_array, train_normal_array, chunksize):
		hot_ctr = 0
		normal_ctr = 0

		i = int(current_process().name)

		print("Running process : ", int(current_process().name), " with pid : ", os.getpid())
		for a, train_tuple in enumerate(train_data):
			lS_i = []
			for b, lS_i_index in enumerate(train_tuple[1]):
				if (b, int(lS_i_index)) in hot_emb_dict[b].keys():
					lS_i.append(hot_emb_dict[b][(b, int(lS_i_index))])
				else:
					break

			if ( len(lS_i) == len(train_tuple[1])):
				lS_i = np.array(lS_i).astype(np.float32)
				train_tuple[1] = lS_i
				train_hot_array[i*chunksize + hot_ctr] = train_tuple
				hot_ctr += 1
			else:
				train_normal_array[i*chunksize + normal_ctr] = train_tuple
				normal_ctr += 1
		print("Process : ", int(current_process().name), " done with hot inputs ", hot_ctr, " and normal inputs ", normal_ctr)
		
	# Input Profiler
	print("Input Profiling Initializing!!\n")

	L = args.hot_emb_gpu_mem
	x = args.ip_sampling_rate
	num_hot_emb = args.hot_emb_gpu_mem // (4 * args.arch_sparse_feature_size)

	print("Available GPU Memory for Hot Emb : ", L / (1024 * 1024), " MB")
	print("Input Sampling Rate for Profiling : ", x, "%")

	# =============================== PROFILING START ======================================
	profiling_begin = time_wrap()

	sample_train_data_len = int((x / 100) * len(train_data))
	print("Training Input Dataset Length (D) : ", len(train_data))
	sampled_train_data = np.random.randint(0, len(train_data), size = sample_train_data_len)
	print("Sampled Training Input Dataset Length (D^) : ", len(sampled_train_data))

	# ================== Skew Table Creation ======================
	skew_table = []
			
	for i in range(len(ln_emb)):
		temp_list = np.zeros((ln_emb[i],3), dtype = int)
		skew_table.append(temp_list)

	# =================== Filling Skew Table Emb Table ======================
	for i in range(len(ln_emb)):
		for j in range(ln_emb[i]):
			skew_table[i][j][0] = i
			
	# =================== Filling Skew Table Emb Index ======================
	for i in range(len(ln_emb)):
		for j in range(ln_emb[i]):
			skew_table[i][j][1] = j	

	# =================== Filling Skew Table Emb Counter ======================
	# Updating Skew table with sampled input profiling data
	for i, sample in enumerate(sampled_train_data):
		X, lS_i, label = train_data[sample]
		for j, lS_i_index in enumerate(lS_i):
			skew_table[j][int(lS_i_index)][2] = skew_table[j][int(lS_i_index)][2] + 1

	# Combining skew table list into a 2D array
	skew_table_array = np.vstack(skew_table) 

	# =================== Sorting Skew Table based on Counter ==============
	skew_table_array = skew_table_array[skew_table_array[:,2].argsort()[::-1]]
			
	# =================== Getting hot embedding entries ====================
	hot_emb_entries = skew_table_array[0:num_hot_emb]

	# =================== Getting Top Emb Dict ==============================
	hot_emb_dict = []
	emb_dict = {}
	for i in range(len(ln_emb)):
		new_emb_dict = copy.deepcopy(emb_dict)
		hot_emb_dict.append(new_emb_dict)

	for i in range(len(hot_emb_entries)):
		hot_emb_dict[hot_emb_entries[i][0]][(hot_emb_entries[i][0], hot_emb_entries[i][1])] = np.float32(i)
	
	len_hot_emb_dict = 0
	for i in range(len(hot_emb_dict)):
		len_hot_emb_dict += len(hot_emb_dict[i])

	del skew_table_array
	print("Hot Emb Dict Size : ", (len_hot_emb_dict * 4 * args.arch_sparse_feature_size) / (1024 ** 2), " MB")
	print("Hot Emb Dict Creation Completed!!")
	
	# ===================== Input Profiling ========================
	print("Starting Input Classification")
	
	num_cores = mp.cpu_count()
	print("Num Cores : ", num_cores)
	chunksize = len(train) // num_cores

	processes = [Process(target = single_process_ip_classification,
						name = "%i" % i,
						args = (train[i*chunksize : (i+1)*chunksize],
								hot_emb_dict,
								train_hot_array,
								train_normal_array,
								chunksize
								)
						)
				for i in range(0, num_cores)]
	
	for process in processes:
		process.start()
	
	for process in processes:
		process.join()
	
	# Removing None elements from both train hot and train normal arrays
	nan_array_hot = pd.isnull(train_hot_array)
	not_nan_array_hot = ~ nan_array_hot
	train_hot_array = train_hot_array[not_nan_array_hot]
	
	nan_array_normal = pd.isnull(train_normal_array)
	not_nan_array_normal = ~ nan_array_normal
	train_normal_array = train_normal_array[not_nan_array_normal]

	print("===================== Input Profiling Stats ==================")
	print("Train Hot Data : ", len(train_hot_array))
	print("Train Normal Data : ", len(train_normal_array))
	print("Total Data : ", len(train_hot_array) + len(train_normal_array))
	print("Percentage : ", (len(train_hot_array) / (len(train_hot_array) + len(train_normal_array))) * 100 )
	print("==============================================================")

	# Closing the shared memories and unlinking
	shm_train_hot.close()
	shm_train_hot.unlink()

	shm_train_normal.close()
	shm_train_normal.unlink()

	profiling_end = time_wrap()
	print("Profiling Time : ", profiling_end - profiling_begin, " s")

	train_hot = np.array(train_hot_array, dtype = object)
	train_normal = np.array(train_normal_array, dtype = object)
	hot_emb_dict = np.array(hot_emb_dict, dtype = object)
	
	if args.data_set == "kaggle":
		np.savez_compressed('./input/kaggle_hot_cold/train_hot.npz', train_hot)
		np.savez_compressed('./input/kaggle_hot_cold/train_normal.npz', train_normal)
		np.savez_compressed('./input/kaggle_hot_cold/hot_emb_dict.npz', hot_emb_dict)
	elif args.data_set == "terabyte":
		np.savez_compressed('./input/terabyte_hot_cold/train_hot.npz', train_hot)
		np.savez_compressed('./input/terabyte_hot_cold/train_normal.npz', train_normal)
		np.savez_compressed('./input/terabyte_hot_cold/hot_emb_dict.npz', hot_emb_dict)
				
	print("Save Hot/Cold Data Completed")
	sys.exit("FAE pre-processing completed!!")