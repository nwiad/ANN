# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm2d, self).__init__()
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
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)
			var = torch.var(input, dim=(0, 2, 3), keepdim=True)
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
			shifted_input = (input - mean) / torch.sqrt(var + self.epsilon)
		else:
			shifted_input = (input - self.running_mean.unsqueeze(-1).unsqueeze(-1)) / torch.sqrt(self.running_var.unsqueeze(-1).unsqueeze(-1) + self.epsilon)
		return shifted_input * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1)
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			# randomly zero out some feature maps (input.shape[1]) instead of some neurons
			return 1 / (1 - self.p) * input * torch.bernoulli(torch.ones(input.shape[0], input.shape[1], device=input.device) * (1 - self.p)).unsqueeze(-1).unsqueeze(-1)
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, batch_norm=True, dropout=True):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		print("dropout before relu")
		self.out_channels = [64, 64]
		self.kernel_size = [3, 3]
		self.maxpool_size = [2, 2]
		self.batch_norm = batch_norm
		self.dropout = dropout
		self.conv1 = nn.Conv2d(3, self.out_channels[0], self.kernel_size[0], padding=(self.kernel_size[0] - 1) // 2)
		self.bn1 = BatchNorm2d(self.out_channels[0])
		self.bn2 = BatchNorm2d(self.out_channels[1])
		self.dropout1 = Dropout(drop_rate)
		self.dropout2 = Dropout(drop_rate)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(self.maxpool_size[0], stride=self.maxpool_size[0])
		self.conv2 = nn.Conv2d(self.out_channels[0], self.out_channels[1], self.kernel_size[1], padding=(self.kernel_size[1] - 1) // 2)
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(self.maxpool_size[1], stride=self.maxpool_size[1])
		self.fc = nn.Linear(32//self.maxpool_size[0]//self.maxpool_size[1]*32//self.maxpool_size[0]//self.maxpool_size[1]*self.out_channels[1], 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.dropout1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.dropout2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		x = x.view(x.shape[0], -1)
		logits = self.fc(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
