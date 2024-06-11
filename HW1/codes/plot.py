import matplotlib.pyplot as plt
import os
def plot(activation, loss, hidden_layers, epochs, alpha, train_axis, test_axis, train_loss, test_loss, train_acc, test_acc):
    log_path = f"res/{hidden_layers}+{activation}+{loss}+{epochs}epochs"
    if alpha > 0:
        log_path += f"+alpha={alpha}"
    if not os.path.exists("res/"):
        os.makedirs("res/")
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(train_axis, train_loss, label='train_loss')
    plt.plot(test_axis, test_loss, label='test_loss')
    plt.legend()
    plt.title('loss')
    plt.subplot(122)
    plt.plot(train_axis, train_acc, label='train_acc')
    plt.plot(test_axis, test_acc, label='test_acc')
    plt.legend()
    plt.title('accuracy')
    plt.suptitle(f"{activation}, {loss}, {hidden_layers} hidden layer") if hidden_layers == 1 else plt.suptitle(f"{activation}, {loss}, {hidden_layers} hidden layers")
    plt.savefig(log_path + '.png')