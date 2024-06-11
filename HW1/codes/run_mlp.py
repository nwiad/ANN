from network import Network
from utils import LOG_INFO
from layers import Selu, Swish, Linear, Gelu, Relu
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net, train_axis, train_loss, train_acc, test_axis, test_loss, test_acc
from load_data import load_mnist_2d

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--activation", type=str, default="Relu")
parser.add_argument("--loss", type=str, default="MSELoss")
parser.add_argument("--hidden_layers", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--alpha", type=float, default=-0.1)
activations = {"Selu": Selu, "Swish": Swish, "Gelu": Gelu, "Relu": Relu}
losses = {"MSELoss": MSELoss, "SoftmaxCrossEntropyLoss": SoftmaxCrossEntropyLoss, "HingeLoss": HingeLoss, "FocalLoss": FocalLoss}
current_activation = parser.parse_args().activation
current_loss = parser.parse_args().loss
current_hidden_layers = parser.parse_args().hidden_layers
current_epochs = parser.parse_args().epochs
assert current_hidden_layers in [1, 2], "hidden layers must be 1 or 2"  
print(f"current activation is: {current_activation}")
print(f"current loss is: {current_loss}")
print(f"current number of hidden layers is: {current_hidden_layers}")
print(f"current number of epochs is: {current_epochs}")
activation_layer = activations[current_activation]
loss_func = losses[current_loss]

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
if current_hidden_layers == 1:
    model.add(Linear('fc1', 784, 128, 0.01))
    model.add(activation_layer('act1'))
    model.add(Linear('fc2', 128, 10, 0.01))
elif current_hidden_layers == 2:
    model.add(Linear('fc1', 784, 128, 0.01))
    model.add(activation_layer('act1'))
    model.add(Linear('fc2', 128, 64, 0.01))
    model.add(activation_layer('act2'))
    model.add(Linear('fc3', 64, 10, 0.01))

loss = loss_func(name='loss', alpha=[parser.parse_args().alpha for _ in range(10)]) if current_loss == "FocalLoss" else loss_func(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': current_epochs, # default = 20
    'disp_freq': 150,
    'test_epoch': 5
}

import time
training_time = 0.0
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    start = time.time()
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], epoch, final=epoch == config['max_epoch'] - 1)
    end = time.time()
    training_time += end - start

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'], epoch + 1)

    if epoch == config['max_epoch'] - 1:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'], epoch + 1, final=True)

with open("results.txt", 'a') as f:
    f.write(f'train_axis={str(train_axis)}\n')
    f.write(f'train_loss={str(train_loss)}\n')
    f.write(f'train_acc={str(train_acc)}\n')
    f.write(f'test_axis={str(test_axis)}\n')
    f.write(f'test_loss={str(test_loss)}\n')
    f.write(f'test_acc={str(test_acc)}\n')
    f.write(f"Training time==[{training_time},{training_time / config['max_epoch']}]\n")
from plot import plot
plot(current_activation, current_loss, current_hidden_layers, current_epochs, parser.parse_args().alpha, train_axis, test_axis, train_loss, test_loss, train_acc, test_acc)
