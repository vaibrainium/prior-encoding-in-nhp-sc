import numpy as np
import torch
from torch.nn.init import normal, constant
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

class FixedLikelihoodNet(nn.Module):
	def __init__(self, n_channel, n_hidden, n_output, dropout=0.5, sigma_init=3, std=0.01):
		super().__init__()
		self.n_channel = n_channel
		self.std = std
		self.n_output = n_output
		self.dropout = dropout
		self.sigma_init = sigma_init

		if not isinstance(n_hidden, (list, tuple)):
			n_hidden = (n_hidden,)
		n_hidden = [i for i in n_hidden if i != 0]
		self.n_hidden = n_hidden

		self.n_likelihood = n_output
		self.likelihood = nn.Parameter(torch.ones(1, 1, 1, self.n_likelihood))

		grid_x = torch.linspace(-1, 1, n_output)
		grid_y = torch.ones(n_output)
		self.register_buffer('grid_x', grid_x)
		self.register_buffer('grid_y', grid_y)
		n_prev = n_channel

		hiddens = OrderedDict()

		for i, n in enumerate(n_hidden):
			prefix = 'layer{}'.format(i)
			hiddens[prefix] = nn.Linear(n_prev, n)
			hiddens[prefix + '_nonlin'] = nn.ReLU()
			if dropout > 0.0:
				hiddens[prefix + '_dropout'] = nn.Dropout(p=dropout, inplace=False)
			n_prev = n
		self.hiddens = nn.Sequential(hiddens)

		self.mu_ro = nn.Linear(n_prev, 1)
		self.register_buffer('bins', (torch.arange(n_output) - n_output // 2).unsqueeze(0))

		self.initialize()

	def forward(self, x):
		x = self.hiddens(x)
		mus = self.mu_ro(x) # This final linear layer maps the hidden features to a single value per sample — the predicted mean of the likelihood distribution.

		grid_x = Variable(self.grid_x) + mus # shifts the grid to center aroung predicted mu

		likelihood = self.likelihood

		grid = torch.stack([grid_x, Variable(self.grid_y).expand_as(grid_x)], dim=-1).unsqueeze(1) # grid_x and grid_y are stacked to form a grid of likelihood values -> 2D array is required for F.grid_sample so grid_y is dummy second dimension
		# likelihood is a fixed Gaussian-looking tensor (1x1x1xn_output). It gets shifted by the grid, producing a per-sample likelihood over bins. Grid_sample warps the original tensor based on the predicted mean — kind of like sliding a heatmap along an axis.
		shifted_likelihood = F.grid_sample(likelihood.expand(x.shape[0], -1, -1, -1), grid, padding_mode='border')

		# remove unneeded dimensions
		return shifted_likelihood.view([-1, shifted_likelihood.shape[-1]])

	def l2_weights(self, avg=True):
		reg = dict(weight=0.0, counts=0)

		def accum(mod):
			if isinstance(mod, nn.Linear):
				# sum of squared weights and count how many weights were added
				reg['weight'] = reg['weight'] + mod.weight.pow(2).sum()
				reg['counts'] = reg['counts'] + mod.weight.numel()
		# apply to every submodule
		self.apply(accum)

		ret = reg['weight']
		if avg:
			ret = ret / reg['counts']
		return ret

	def l2_weights_per_layer(self):
		reg = dict(weight=0.0, counts=0)

		def accum(mod):
			if isinstance(mod, nn.Linear):
				reg['weight'] = reg['weight'] + mod.weight.pow(2).mean()
				reg['counts'] = reg['counts'] + 1

		self.apply(accum)

		return reg['weight'] / reg['counts']

	def initialize(self):
		# initialize normalized weights for each layer
		def fn(mod):
			if isinstance(mod, nn.Linear):
				normal(mod.weight, std=self.std)
				constant(mod.bias, 0)

		self.apply(fn)
		self.likelihood.data.copy_(
			-(torch.arange(self.n_likelihood).view(1, 1, 1, -1) - self.n_likelihood // 2).pow(2) / 2 / self.sigma_init ** 2)

