import os
import numpy as np
from matplotlib import pyplot as plt

data_dir = './datasets'
def read_clean_dataset(summary=False):
    """Load clean dataset."""
    data = np.load(os.path.join(data_dir, 'full.npz'))
    features = data['features']
    labels = data['labels']
    if summary:
        print('Loaded clean data:')
        print('Data has shape = {}, contains {} unique labels'.format(
            features.shape, len(np.unique(labels))))
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(features, aspect='auto')
        ax.set_xlabel('feature dim')
        ax.set_ylabel(r'$x_{i}$')
        ax.set_title('Clean Features')
        cbar = fig.colorbar(im)
        plt.show()
    return features, labels

def read_corrupted_dataset(summary=False):
    """Load corrupted dataset."""
    data = np.load(os.path.join(data_dir, 'corrupted_series.npz'))
    features = data['features']
    length = data['length']
    if summary:
        print('Loaded corrupted data:')
        print('Data has shape = {}, average sequence length = {:0.2f}'.format(
            features.shape, np.mean(length)))
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(121)
        im = ax.imshow(features, aspect='auto')
        ax.set_xlabel('feature dim')
        ax.set_ylabel(r'$x_{i}$')
        ax.set_title('Corrupted Features')
        ax = fig.add_subplot(122)
        ax.hist(length, bins=20)
        ax.set_title('Histogram of Sequence Length')
        ax.set_xlabel('length')
        ax.set_ylabel('count')
        plt.show()
    return features, length

def one_hot(y):
    """
    Convert dataset labels to one-hot.
    Method will not work for datasets with non-integer/index labels.
    """
    num_class = len(np.unique(y))
    y_one_hot = np.zeros((len(y), num_class), dtype=np.int32)
    for i, y_i in zip(range(len(y)), y):
        y_one_hot[i,y_i] = 1
    return y_one_hot
