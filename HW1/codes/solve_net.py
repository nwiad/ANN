from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np

train_axis, train_loss, train_acc = [], [], []
test_axis, test_loss, test_acc = [], [], []

def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq, step, final=False):

    iter_counter = 0
    loss_list = []
    acc_list = []

    # the length of data_iterator
    length = len(inputs) // batch_size

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

        if final and iter_counter == length:
            with open("results.txt", "a") as f:
                f.write(f"train_loss={np.mean(loss_list)}\ntrain_acc={np.mean(acc_list)}\n")

        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            train_axis.append(step + iter_counter / length)
            train_loss.append(np.mean(loss_list))
            train_acc.append(np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)
        



def test_net(model, loss, inputs, labels, batch_size, step, final=False):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    test_axis.append(step)
    test_loss.append(np.mean(loss_list))
    test_acc.append(np.mean(acc_list))
    LOG_INFO(msg)
    if final:
        with open("results.txt", "a") as f:
            f.write(f"test_loss={np.mean(loss_list)}\ntest_acc={np.mean(acc_list)}\n")
