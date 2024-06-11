# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.momentum = 0.9
		self.epsilon = 1e-5

		# Parameters
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter


	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mean = torch.mean(input, dim=0, keepdim=True)
			var = torch.var(input, dim=0, keepdim=True)
			self.running_mean = self.running_mean * self.momentum + mean * (1 - self.momentum)
			self.running_var = self.running_var * self.momentum + var * (1 - self.momentum)
			shifted_input = (input - mean) / torch.sqrt(var + self.epsilon)
		else:
			shifted_input = (input - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
		return shifted_input * self.weight + self.bias
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			return 1 / (1 - self.p) * input * torch.bernoulli(torch.ones_like(input) * (1 - self.p))
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, batch_norm=True, dropout=True):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.hidden_dim = 1200
		self.batch_norm = batch_norm
		self.dropout = dropout
		self.fc1 = nn.Linear(3 * 32 * 32, self.hidden_dim)
		if batch_norm:
			self.bn = BatchNorm1d(self.hidden_dim)
		self.relu = nn.ReLU()
		if dropout:
			self.dropout = Dropout(drop_rate)
		self.fc2 = nn.Linear(self.hidden_dim, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = self.fc1(x)
		if self.batch_norm:
			x = self.bn(x)
		x = self.relu(x)
		if self.dropout:
			x = self.dropout(x)
		logits = self.fc2(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
