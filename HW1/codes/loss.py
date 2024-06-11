from __future__ import division
import numpy as np


class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        squared_delta = np.square(input - target)
        squared_norm = np.sum(squared_delta, axis=1)
        return np.mean(squared_norm)
        # TODO END

    def backward(self, input, target):
		# TODO START
        return 2 * (input - target) / input.shape[0]
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        exp_x = np.exp(input)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        softmax_x = exp_x / sum_exp_x # softmax(x) element-wise
        ln_x = np.log(softmax_x) # log_softmax(x) element-wise
        t_ln_x = np.multiply(ln_x, target) # muitlple ln_x and t element-wise
        neg_sum_t_ln_x = -np.sum(t_ln_x, axis=1)
        return np.mean(neg_sum_t_ln_x)
        # TODO END

    def backward(self, input, target):
        # TODO START
        exp_x = np.exp(input)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        softmax_x = exp_x / sum_exp_x
        return (softmax_x - target) / input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        correct_labels = input[np.arange(input.shape[0]), np.argmax(target, axis=1)]
        margins = np.maximum(0, input - correct_labels[:, np.newaxis] + self.margin)
        margins[np.arange(input.shape[0]), np.argmax(target, axis=1)] = 0
        self.margins = margins
        return np.mean(np.sum(margins, axis=1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        # 计算梯度
        grad = np.zeros_like(input)
        grad[self.margins > 0] = 1
        grad[np.arange(input.shape[0]), np.argmax(target, axis=1)] -= np.sum(grad, axis=1)
        grad /= input.shape[0]
        return grad
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        self.alpha = np.array(alpha) if alpha else np.array([0.1 for _ in range(10)])
        self.gamma = gamma
        

    def forward(self, input, target):
        # TODO START
        exp_x = np.exp(input)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        softmax_x = exp_x / sum_exp_x # softmax(x) element-wise
        elem = (self.alpha * target + (1 - self.alpha) * (1 - target)) * np.power(1 - softmax_x, self.gamma) * target * np.log(softmax_x)
        return -np.mean(np.sum(elem, axis=1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        exp_x = np.exp(input)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        softmax_x = exp_x / sum_exp_x
        A = (self.alpha * target + (1 - self.alpha) * (1 - target)) * target * (np.power(1 - softmax_x, self.gamma) - self.gamma * softmax_x * np.power(1 - softmax_x, self.gamma - 1) * np.log(softmax_x))
        return (softmax_x * np.sum(A, axis=1, keepdims=True) - A) / input.shape[0]
        # TODO END
