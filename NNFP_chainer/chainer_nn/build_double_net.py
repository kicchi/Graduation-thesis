from utils import memoize, WeightsParser
from rdkit_utils import smiles_to_fps
from build_convnet import build_convnet_fingerprint_fun
from build_vanilla_net import build_fingerprint_deep_net

import numpy as np

def build_double_convnet_fingerprint_fun(**kwargs):
	fp_fun1, parse1 = build_convnet_fingerprint_fun(**kwargs)
	fp_fun2, parse2 = build_convnet_fingerprint_fun(**kwargs)

	def double_fingerprint_fun(weights, smiles_tuple):
		smiles1, smiles2 = zip(*smiles_tuple)
		fp1 = fp_fun1(weights, smiles1)
		fp2 = fp_fun2(weights, smiles2)

		return zip(fp1, fp2)

	commbined_parser = WeightsParser()
	commbined_parser.add_weights('weights1', len(parser1))
	commbined_parser.add_weights('weights2', len(parser2))

	return double_fingerprint_fun, commbined_parser

def build_double_conv_deep_net(conv_params, net_params, fp_l2_penalty=0.0):
	"""Returns loss_fun(all_weights, smiles, targets), pred_fun, commbined_parser."""
	conv_fp_func, conv_parser = build_double_convnet_fingerprint_fun(**conv_paras)
	return build_fingerprint_deep_net(net_params, conv_fp_func, conv_parser, fp_l2_penalty)
