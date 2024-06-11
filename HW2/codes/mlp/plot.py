# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
def plot(axis, train_acc, train_loss, val_acc, val_loss, batch_size, drop_rate, learning_rate):
    if not os.path.exists('res'):
        os.mkdir('res')
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(axis, train_acc, label='train_acc')
    plt.plot(axis, val_acc, label='val_acc')
    plt.legend()
    plt.title('accuracy')
    plt.subplot(122)
    plt.plot(axis, train_loss, label='train_loss')
    plt.plot(axis, val_loss, label='val_loss')
    plt.legend()
    plt.title('loss')
    plt.savefig(f'res/mlp_{batch_size}_{drop_rate}_{learning_rate}.png')