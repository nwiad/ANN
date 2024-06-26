import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input = self._saved_tensor
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output = self._saved_tensor
		return grad_output * output * (1 - output)

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)
        self.lambda_ = 1.0507
        self.alpha = 1.6732

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        mask = input > 0
        pos = self.lambda_ * input * mask
        neg = self.lambda_ * self.alpha * (np.exp(input) - 1) * (1 - mask)
        return pos + neg
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        mask = input > 0
        pos = self.lambda_ * mask
        neg = self.lambda_ * self.alpha * np.exp(input) * (1 - mask)
        return grad_output * (pos + neg)
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        output = input / (1 + np.exp(-input))
        self._saved_for_backward(output) # 保存输出
        self.swish_input = input # 额外保存输入
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self.swish_input
        output = self._saved_tensor
        return grad_output * (output + (1 - output) / (1 + np.exp(-input)))
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        Tanh = np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * np.power(input, 3)))
        self._saved_for_backward(Tanh) # 保存 Tanh
        self.gelu_input = input # 额外保存输入
        return input * 0.5 * (1 + Tanh)
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        input = self.gelu_input
        Tanh = self._saved_tensor
        return grad_output * (
            0.5 * (1 + Tanh) + 
            input * 0.5 * (1 - np.power(Tanh, 2)) * np.sqrt(2 / np.pi) * (1 + 0.134145 * np.power(input, 2))
            )
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return input.dot(self.W) + self.b # [batch_size, in_num] * [in_num, out_num] = [batch_size, out_num]
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        self.grad_W = input.T.dot(grad_output) # transpose([batch_size, in_num]) * [batch_size, out_num] = [in_num, out_num]
        self.grad_b = np.sum(grad_output, axis=0) # [batch_size, out_num]
        return grad_output.dot(self.W.T) # [batch_size, out_num] * transpose([in_num, out_num]) = [batch_size, in_num]
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
