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
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=DeprecationWarning)
#import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler

exc = getattr(builtins, "IOError", "FileNotFoundError")

class LRPolicyScheduler(_LRScheduler):
	def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
		self.num_warmup_steps = num_warmup_steps
		self.decay_start_step = decay_start_step
		self.decay_end_step = decay_start_step + num_decay_steps
		self.num_decay_steps = num_decay_steps

		if self.decay_start_step < self.num_warmup_steps:
			sys.exit("Learning rate warmup must finish before the decay starts")

		super(LRPolicyScheduler, self).__init__(optimizer)

	def get_lr(self):
		step_count = self._step_count
		if step_count < self.num_warmup_steps:
			# warmup
			scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
			lr = [base_lr * scale for base_lr in self.base_lrs]
			self.last_lr = lr
		elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
			# decay
			decayed_steps = step_count - self.decay_start_step
			scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
			min_lr = 0.0000001
			lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
			self.last_lr = lr
		else:
			if self.num_decay_steps > 0:
				# freeze at last, either because we're after decay
				# or because we're between warmup and decay
				lr = self.last_lr
			else:
				# do not adjust
				lr = self.base_lrs
		return lr

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
	def create_mlp(self, ln, sigmoid_layer):
		# build MLP layer by layer
		layers = nn.ModuleList()
		for i in range(0, ln.size - 1):
			n = ln[i]
			m = ln[i + 1]

			# construct fully connected operator
			LL = nn.Linear(int(n), int(m), bias=True)

			# initialize the weights
			# with torch.no_grad():
			# custom Xavier input, output or two-sided fill
			mean = 0.0  # std_dev = np.sqrt(variance)
			std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
			W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
			std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
			bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
			# approach 1
			LL.weight.data = torch.tensor(W, requires_grad=True)
			LL.bias.data = torch.tensor(bt, requires_grad=True)
			# approach 2
			# LL.weight.data.copy_(torch.tensor(W))
			# LL.bias.data.copy_(torch.tensor(bt))
			# approach 3
			# LL.weight = Parameter(torch.tensor(W),requires_grad=True)
			# LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
			layers.append(LL)

			# construct sigmoid or relu operator
			if i == sigmoid_layer:
				layers.append(nn.Sigmoid())
			else:
				layers.append(nn.ReLU())

		# approach 1: use ModuleList
		# return layers
		# approach 2: use Sequential container to wrap all layers
		return torch.nn.Sequential(*layers)

	def create_emb(self, m, ln):
		emb_l = nn.ModuleList()
		for i in range(0, ln.size):
			n = ln[i]
			# construct embedding operator
			if self.qr_flag and n > self.qr_threshold:
				EE = QREmbeddingBag(n, m, self.qr_collisions,
					operation=self.qr_operation, mode="sum", sparse=True)
			elif self.md_flag:
				base = max(m)
				_m = m[i] if n > self.md_threshold else base
				EE = PrEmbeddingBag(n, _m, base)
				# use np initialization as below for consistency...
				W = np.random.uniform(
					low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
				).astype(np.float32)
				EE.embs.weight.data = torch.tensor(W, requires_grad=True)

			else:
				EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

				# initialize embeddings
				# nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
				W = np.random.uniform(
					low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
				).astype(np.float32)
				# approach 1
				EE.weight.data = torch.tensor(W, requires_grad=True)
				# approach 2
				# EE.weight.data.copy_(torch.tensor(W))
				# approach 3
				# EE.weight = Parameter(torch.tensor(W),requires_grad=True)

			emb_l.append(EE)

		return emb_l

	def __init__(
		self,
		m_spa=None,
		ln_emb=None,
		ln_bot=None,
		ln_top=None,
		arch_interaction_op=None,
		arch_interaction_itself=False,
		sigmoid_bot=-1,
		sigmoid_top=-1,
		sync_dense_params=True,
		loss_threshold=0.0,
		ndevices=-1,
		qr_flag=False,
		qr_operation="mult",
		qr_collisions=0,
		qr_threshold=200,
		md_flag=False,
		md_threshold=200,
	):
		super(DLRM_Net, self).__init__()

		if (
			(m_spa is not None)
			and (ln_emb is not None)
			and (ln_bot is not None)
			and (ln_top is not None)
			and (arch_interaction_op is not None)
		):

			# save arguments
			self.ndevices = ndevices
			self.output_d = 0
			self.parallel_model_batch_size = -1
			self.parallel_model_is_not_prepared = True
			self.arch_interaction_op = arch_interaction_op
			self.arch_interaction_itself = arch_interaction_itself
			self.sync_dense_params = sync_dense_params
			self.loss_threshold = loss_threshold
			# create variables for QR embedding if applicable
			self.qr_flag = qr_flag
			if self.qr_flag:
				self.qr_collisions = qr_collisions
				self.qr_operation = qr_operation
				self.qr_threshold = qr_threshold
			# create variables for MD embedding if applicable
			self.md_flag = md_flag
			if self.md_flag:
				self.md_threshold = md_threshold
			
			self.emb_l = self.create_emb(m_spa, ln_emb)
			print("EMB : ", ln_emb)
			self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
			self.bot_l = self.bot_l.to("cuda:0")
			self.top_l = self.create_mlp(ln_top, sigmoid_top)
			self.top_l = self.top_l.to("cuda:0")

	def apply_mlp(self, x, layers):
		# approach 1: use ModuleList
		# for layer in layers:
		#     x = layer(x)
		# return x
		# approach 2: use Sequential container to wrap all layers
		return layers(x)

	def apply_emb(self, lS_o, lS_i, emb_l):
		# WARNING: notice that we are processing the batch at once. We implicitly
		# assume that the data is laid out such that:
		# 1. each embedding is indexed with a group of sparse indices,
		#   corresponding to a single lookup
		# 2. for each embedding the lookups are further organized into a batch
		# 3. for a list of embedding tables there is a list of batched lookups

		ly = []
		# for k, sparse_index_group_batch in enumerate(lS_i):
		for k in range(len(lS_i)):
			sparse_index_group_batch = lS_i[k]
			sparse_offset_group_batch = lS_o[k]

			# embedding lookup
			# We are using EmbeddingBag, which implicitly uses sum operator.
			# The embeddings are represented as tall matrices, with sum
			# happening vertically across 0 axis, resulting in a row vector
			E = emb_l[k]
			V = E(sparse_index_group_batch, sparse_offset_group_batch)

			ly.append(V)

		# print(ly)
		return ly

	def interact_features(self, x, ly):
		if self.arch_interaction_op == "dot":
			# concatenate dense and sparse features
			(batch_size, d) = x.shape
			T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
			# perform a dot product
			Z = torch.bmm(T, torch.transpose(T, 1, 2))
			# append dense feature with the interactions (into a row vector)
			# approach 1: all
			# Zflat = Z.view((batch_size, -1))
			# approach 2: unique
			_, ni, nj = Z.shape
			# approach 1: tril_indices
			# offset = 0 if self.arch_interaction_itself else -1
			# li, lj = torch.tril_indices(ni, nj, offset=offset)
			# approach 2: custom
			offset = 1 if self.arch_interaction_itself else 0
			li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
			lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
			Zflat = Z[:, li, lj]
			# concatenate dense features and interactions
			R = torch.cat([x] + [Zflat], dim=1)
		elif self.arch_interaction_op == "cat":
			# concatenation features (into a row vector)
			R = torch.cat([x] + ly, dim=1)
		else:
			sys.exit(
				"ERROR: --arch-interaction-op="
				+ self.arch_interaction_op
				+ " is not supported"
			)

		return R

	def forward(self, dense_x, lS_o, lS_i):
		return self.sequential_forward(dense_x, lS_o, lS_i)
	
	def sequential_forward(self, dense_x, lS_o, lS_i):
		# process dense features (using bottom mlp), resulting in a row vector
		x = self.apply_mlp(dense_x, self.bot_l)

		# process sparse features(using embeddings), resulting in a list of row vectors
		ly = self.apply_emb(lS_o, lS_i, self.emb_l)
		ly = torch.stack(ly)

		# Moving ly across GPU
		ly = ly.to("cuda:0")
		ly = list(ly)

		# interact features (dense and sparse)
		z = self.interact_features(x, ly)

		# obtain probability of a click (using top mlp)
		p = self.apply_mlp(z, self.top_l)

		# clamp output if needed
		if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
			z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
		else:
			z = p

		return z	

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
	# embedding table options
	parser.add_argument("--md-flag", action="store_true", default=False)
	parser.add_argument("--md-threshold", type=int, default=200)
	parser.add_argument("--md-temperature", type=float, default=0.3)
	parser.add_argument("--md-round-dims", action="store_true", default=False)
	parser.add_argument("--qr-flag", action="store_true", default=False)
	parser.add_argument("--qr-threshold", type=int, default=200)
	parser.add_argument("--qr-operation", type=str, default="mult")
	parser.add_argument("--qr-collisions", type=int, default=4)
	# activations and loss
	parser.add_argument("--activation-function", type=str, default="relu")
	parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
	parser.add_argument(
		"--loss-weights", type=dash_separated_floats, default="1.0-1.0")  # for wbce
	parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
	parser.add_argument("--round-targets", type=bool, default=False)
	# data
	parser.add_argument("--data-size", type=int, default=1)
	parser.add_argument("--num-batches", type=int, default=0)
	parser.add_argument(
		"--data-generation", type=str, default="random"
	)  # synthetic or dataset
	parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
	parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte or kaggle_tutorial
	parser.add_argument("--raw-data-file", type=str, default="")
	parser.add_argument("--processed-data-file", type=str, default="")
	# ======================= Add processed data file for tutorial ======================
	parser.add_argument("--processed-data-file-tutorial", type=str, default="")
	# ===================================================================================
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
	# training
	parser.add_argument("--mini-batch-size", type=int, default=1)
	parser.add_argument("--nepochs", type=int, default=1)
	parser.add_argument("--learning-rate", type=float, default=0.01)
	parser.add_argument("--print-precision", type=int, default=5)
	parser.add_argument("--numpy-rand-seed", type=int, default=123)
	parser.add_argument("--sync-dense-params", type=bool, default=True)
	# inference
	parser.add_argument("--inference-only", action="store_true", default=False)
	# onnx
	parser.add_argument("--save-onnx", action="store_true", default=False)
	# gpu
	parser.add_argument("--use-gpu", action="store_true", default=True)
	# debugging and profiling
	parser.add_argument("--print-freq", type=int, default=1)
	parser.add_argument("--test-freq", type=int, default=-1)
	parser.add_argument("--test-mini-batch-size", type=int, default=-1)
	parser.add_argument("--test-num-workers", type=int, default=-1)
	parser.add_argument("--print-time", action="store_true", default=False)
	parser.add_argument("--debug-mode", action="store_true", default=False)
	parser.add_argument("--enable-profiling", action="store_true", default=False)
	parser.add_argument("--plot-compute-graph", action="store_true", default=False)
	# store/load model
	parser.add_argument("--save-model", type=str, default="")
	parser.add_argument("--load-model", type=str, default="")
	# mlperf logging (disables other output and stops early)
	parser.add_argument("--mlperf-logging", action="store_true", default=False)
	# stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
	parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
	# stop at target AUC Terabyte (no subsampling) 0.8025
	parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
	parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
	parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
	# LR policy
	parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
	parser.add_argument("--lr-decay-start-step", type=int, default=0)
	parser.add_argument("--lr-num-decay-steps", type=int, default=0)
	# Embedding Access Characterization
	parser.add_argument("--emb-characterization", type=int, default=0)
	args = parser.parse_args()

	if args.mlperf_logging:
		print('command line args: ', json.dumps(vars(args)))

	### some basic setup ###
	np.random.seed(args.numpy_rand_seed)
	np.set_printoptions(precision=args.print_precision)
	torch.set_printoptions(precision=args.print_precision)
	torch.manual_seed(args.numpy_rand_seed)

	if (args.test_mini_batch_size < 0):
		# if the parameter is not set, use the training batch size
		args.test_mini_batch_size = args.mini_batch_size
	if (args.test_num_workers < 0):
		# if the parameter is not set, use the same parameter for training
		args.test_num_workers = args.num_workers

	use_gpu = args.use_gpu and torch.cuda.is_available()
	if use_gpu:
		torch.cuda.manual_seed_all(args.numpy_rand_seed)
		torch.backends.cudnn.deterministic = True
		device = torch.device("cuda", 0)
		ngpus = torch.cuda.device_count()  # 1
		print("Running DLRM Baseline")
		print("Using CPU and {} GPU(s)...".format(ngpus))
	else:
		device = torch.device("cpu")
		print("Using CPU...")

	### prepare training data ###
	ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
	# input data
	if (args.data_generation == "dataset"):

		if (args.data_set == "kaggle_tutorial"):
			# ============================== Loading processed data =======================
			print("Loading pre-processed kaggle tutorial Data")

			train = np.load(args.processed_data_file_tutorial, allow_pickle = True)
			train = train['arr_0']
			train = train.tolist()
			print("Length Train Data : ", len(train))

			train_ld = dp.load_criteo_preprocessed_tutorial_data_and_loaders(args, train)

			ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
			m_den = ln_bot[0]
			nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
		
		else:
			train_data, train_ld, test_data, test_ld = \
				dp.make_criteo_data_and_loaders(args)
			nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
			nbatches_test = len(test_ld)

			ln_emb = train_data.counts
			# enforce maximum limit on number of vectors per embedding
			if args.max_ind_range > 0:
				ln_emb = np.array(list(map(
					lambda x: x if x < args.max_ind_range else args.max_ind_range,
					ln_emb
				)))
			m_den = train_data.m_den
			ln_bot[0] = m_den
	else:
		# input and target at random
		ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
		m_den = ln_bot[0]
		train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
		nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

	### parse command line arguments ###
	m_spa = args.arch_sparse_feature_size
	num_fea = ln_emb.size + 1  # num sparse + num dense features
	m_den_out = ln_bot[ln_bot.size - 1]
	if args.arch_interaction_op == "dot":
		# approach 1: all
		# num_int = num_fea * num_fea + m_den_out
		# approach 2: unique
		if args.arch_interaction_itself:
			num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
		else:
			num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
	elif args.arch_interaction_op == "cat":
		num_int = num_fea * m_den_out
	else:
		sys.exit(
			"ERROR: --arch-interaction-op="
			+ args.arch_interaction_op
			+ " is not supported"
		)
	arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
	ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

	# sanity check: feature sizes and mlp dimensions must match
	if m_den != ln_bot[0]:
		sys.exit(
			"ERROR: arch-dense-feature-size "
			+ str(m_den)
			+ " does not match first dim of bottom mlp "
			+ str(ln_bot[0])
		)
	if args.qr_flag:
		if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
			sys.exit(
				"ERROR: 2 arch-sparse-feature-size "
				+ str(2 * m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
				+ " (note that the last dim of bottom mlp must be 2x the embedding dim)"
			)
		if args.qr_operation != "concat" and m_spa != m_den_out:
			sys.exit(
				"ERROR: arch-sparse-feature-size "
				+ str(m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
			)
	else:
		if m_spa != m_den_out:
			sys.exit(
				"ERROR: arch-sparse-feature-size "
				+ str(m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
			)
	if num_int != ln_top[0]:
		sys.exit(
			"ERROR: # of feature interactions "
			+ str(num_int)
			+ " does not match first dimension of top mlp "
			+ str(ln_top[0])
		)

	# assign mixed dimensions if applicable
	if args.md_flag:
		m_spa = md_solver(
			torch.tensor(ln_emb),
			args.md_temperature,  # alpha
			d0=m_spa,
			round_dim=args.md_round_dims
		).tolist()

	# test prints (model arch)
	if args.debug_mode:
		print("model arch:")
		print(
			"mlp top arch "
			+ str(ln_top.size - 1)
			+ " layers, with input to output dimensions:"
		)
		print(ln_top)
		print("# of interactions")
		print(num_int)
		print(
			"mlp bot arch "
			+ str(ln_bot.size - 1)
			+ " layers, with input to output dimensions:"
		)
		print(ln_bot)
		print("# of features (sparse and dense)")
		print(num_fea)
		print("dense feature size")
		print(m_den)
		print("sparse feature size")
		print(m_spa)
		print(
			"# of embeddings (= # of sparse features) "
			+ str(ln_emb.size)
			+ ", with dimensions "
			+ str(m_spa)
			+ "x:"
		)
		print(ln_emb)

		print("data (inputs and targets):")
		for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
			# early exit if nbatches was set by the user and has been exceeded
			if nbatches > 0 and j >= nbatches:
				break

			print("mini-batch: %d" % j)
			print(X.detach().cpu().numpy())
			# transform offsets to lengths when printing
			print(
				[
					np.diff(
						S_o.detach().cpu().tolist() + list(lS_i[i].shape)
					).tolist()
					for i, S_o in enumerate(lS_o)
				]
			)
			print([S_i.detach().cpu().tolist() for S_i in lS_i])
			print(T.detach().cpu().numpy())

	ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

	### construct the neural network specified above ###
	# WARNING: to obtain exactly the same initialization for
	# the weights we need to start from the same random seed.
	# np.random.seed(args.numpy_rand_seed)
	dlrm = DLRM_Net(
		m_spa,
		ln_emb,
		ln_bot,
		ln_top,
		arch_interaction_op=args.arch_interaction_op,
		arch_interaction_itself=args.arch_interaction_itself,
		sigmoid_bot=-1,
		sigmoid_top=ln_top.size - 2,
		sync_dense_params=args.sync_dense_params,
		loss_threshold=args.loss_threshold,
		ndevices=ndevices,
		qr_flag=args.qr_flag,
		qr_operation=args.qr_operation,
		qr_collisions=args.qr_collisions,
		qr_threshold=args.qr_threshold,
		md_flag=args.md_flag,
		md_threshold=args.md_threshold,
	)
	# test prints
	if args.debug_mode:
		print("initial parameters (weights and bias):")
		for param in dlrm.parameters():
			print(param.detach().cpu().numpy())
		# print(dlrm)

	# specify the loss function
	if args.loss_function == "mse":
		loss_fn = torch.nn.MSELoss(reduction="mean")
	elif args.loss_function == "bce":
		loss_fn = torch.nn.BCELoss(reduction="mean")
	elif args.loss_function == "wbce":
		loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
		loss_fn = torch.nn.BCELoss(reduction="none")
	else:
		sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

	if not args.inference_only:
		# specify the optimizer algorithm
		optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
		lr_scheduler = LRPolicyScheduler(optimizer, args.lr_num_warmup_steps, args.lr_decay_start_step,
										 args.lr_num_decay_steps)

	### main loop ###
	def time_wrap(use_gpu):
		if use_gpu:
			torch.cuda.synchronize()
		return time.time()

	# =============================== SAMPLING FOR BIGGEST EMB TABLE - EMB[args.emb_characterization] ======================================
			
	print("Train data length : ", len(train))
			
	total_access = np.zeros((ln_emb[int(args.emb_characterization)],2), dtype = int)
	sample_access = np.zeros((ln_emb[int(args.emb_characterization)],2), dtype = int)

	# =================== Filling Skew Table ======================
	for i in range(len(total_access)):
		total_access[i][0] = i
		sample_access[i][0] = i

	for i, (X, lS_i, T) in enumerate(train):
		lS_i_index = lS_i[int(args.emb_characterization)]
		total_access[int(lS_i_index)][1] = total_access[int(lS_i_index)][1] + 1

	seeds = np.random.randint(0, len(train), size = int(0.05 * len(train)))
	print("Sample data length : ", len(seeds))

	for i, seed in enumerate(seeds):
		X, lS_i, T = train[seed]
		lS_i_index = lS_i[args.emb_characterization]
		sample_access[int(lS_i_index)][1] = sample_access[int(lS_i_index)][1] + 1
				
	#indices = []
	#for i, _ in enumerate(total_access):
	#	if (total_access[i][1] < 2):
	#		indices.append(i)

	#indices = np.array(indices)

	#total_access = np.delete(total_access, indices, axis = 0)
	sorted_emb = np.sort(total_access[:,1])[::-1]
	#sorted_emb_sample = np.sort(sample_access[:,1])[::-1]
	#sample_indices = []
	#for i, _ in enumerate(sample_access):
	#	if (sample_access[i][1] < 2):
	#		sample_indices.append(i)

	#sample_access = np.delete(sample_access, sample_indices, axis = 0)

	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])

	ax.plot(total_access[:,0], sorted_emb, label = 'Original Data', linewidth = 0.5, color = 'b')
	#ax.plot(sample_access[:,0], sorted_emb_sample, label = 'Sampled Data', linewidth = 0.5, color = 'g')
	plt.legend(fontsize = 14, ncol = 2, loc = "upper right", frameon = True, edgecolor='black')
	ax.set_yscale('log')
	plt.xlabel('Indices', fontsize = 16)
	plt.ylabel('Number of Accesses\n(log scale)', fontsize = 16)
	plt.savefig("./input/kaggle/access_freq.png", dpi=200, format='png', bbox_inches='tight')