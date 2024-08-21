# read the data from ../data/training.log and plot the loss and accuracy curves against the number of steps
# the data is stored in the following format:
# Step      1: Total number of trainable weights: 159696
# Step      1: Ran 1 train steps in 35.59 secs
# Step      1: train CrossEntropyLossWithLogSoftmax |  4.00774622
# Step      1: eval  CrossEntropyLossWithLogSoftmax |  3.66965970
# Step      1: eval                CategoryAccuracy |  0.03253174

import matplotlib.pyplot as plt
import numpy as np


def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        metrics = {}
        for line in lines:
            if line.startswith('Step') and not line.startswith('Step      1: Total') and not line.startswith('Step      1: Ran'):
                parts = line.split('|')
                step = int(parts[0].split()[1].split(':')[0])

                if 'train' in line and 'CrossEntropyLossWithLogSoftmax' in line:
                    metrics[step] = {}
                    metrics[step]['train_loss'] = float(parts[1].strip())

                elif 'eval' in line and 'CrossEntropyLossWithLogSoftmax' in line:
                    metrics[step]['eval_loss'] = float(parts[1].strip())

                elif 'eval' in line and 'CategoryAccuracy' in line:
                    metrics[step]['eval_accuracy'] = float(parts[1].strip())

    return metrics


def plot_loss(steps, train_losses, eval_losses):
    plt.figure()
    plt.plot(steps, train_losses, label='train')
    plt.plot(steps, eval_losses, label='eval')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps')
    plt.legend()
    plt.show()


def plot_accuracy(steps, eval_accuracies):
    plt.figure()
    plt.plot(steps, eval_accuracies)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Steps')
    plt.show()


if __name__ == '__main__':
    data = read_data('../data/training.log')
    steps = list(data.keys())
    train_losses = [data[step]['train_loss'] for step in steps]
    eval_losses = [data[step]['eval_loss'] for step in steps]
    eval_accuracies = [data[step]['eval_accuracy'] for step in steps]

    plot_loss(steps, train_losses, eval_losses)
    plot_accuracy(steps, eval_accuracies)
