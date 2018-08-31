import os
import numpy as np
import sklearn.metrics
from keras.datasets import mnist
from IncrementalNearestNeighbor import IncrementalNearestNeighbor
from Queue import Queue
from Reservoir import Reservoir
from OnlineKMeans import OnlineKMeans
import matplotlib.pyplot as plt


def shuffle(X_train, y_train):
    """
    Shuffle training data.
    :param X_train: training data
    :param y_train: training labels
    :return: shuffled data
    """
    np.random.seed(111)
    num_pts = y_train.shape[0]
    ind = np.linspace(0, num_pts - 1, num_pts)
    np.random.shuffle(ind)
    ind = ind.astype(int)
    X_train = X_train[ind]
    y_train = y_train[ind]
    return X_train, y_train


def load_mnist():
    """
    Load MNIST dataset from Keras.
    :return: training and test data and labels
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    X_train /= 255
    X_test /= 255
    return X_train, y_train, X_test, y_test


def get_models(input_shape, dataset_size, num_classes, capacity, cuda_device=None):
    """
    Return list with each model in it.
    :param input_shape: feature length
    :param dataset_size: number of training samples
    :param num_classes: number of classes
    :param capacity: maximum clusters stored per class
    :param cuda_device: None for CPU, else GPU number
    :return: dictionary of model objects
    """
    model_dict = {}
    model_dict['nearest_neighbor'] = IncrementalNearestNeighbor(input_shape, dataset_size, cuda_device=cuda_device)
    model_dict['queue'] = Queue(input_shape, num_classes, capacity, cuda_device=cuda_device)
    model_dict['reservoir'] = Reservoir(input_shape, num_classes, capacity, cuda_device=cuda_device)
    model_dict['online_kmeans'] = OnlineKMeans(input_shape, num_classes, capacity, cuda_device=cuda_device)
    return model_dict


def plot_results(results_dict, eval_num, dataset_size):
    """
    Make plots comparing different models.
    :param results_dict: dictionary with lists of accuracies for each model
    :param eval_num: how often each model was evaluated
    :param dataset_size: total number of training samples
    :return:
    """
    x_vals = np.arange(0, dataset_size, eval_num)

    plt.figure()
    for i in results_dict:
        plt.plot(x_vals, results_dict[i], label=i, linewidth=2)
    plt.xlabel('Sample', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()


def main():
    # set cuda device for computations (None for CPU, else GPU number)
    cuda_device = 0

    # what percentage of all the data would you like the capacity set to?
    capacity_percentage = 10

    # how often during streaming would you like to evaluate?
    eval_num = 5000

    # restrict to single GPU
    if cuda_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # laod dataset and get parameters
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, y_train = shuffle(X_train, y_train)
    input_shape = X_train.shape[1]
    dataset_size = X_train.shape[0]
    num_pts = X_train.shape[0]
    num_classes = 10
    capacity = int(np.floor(X_train.shape[0] / num_classes) * (capacity_percentage / 100))
    print('Capacity Per Class: ', capacity * num_classes)

    # get dictionary of model objects
    models = get_models(input_shape, dataset_size, num_classes, capacity, cuda_device=cuda_device)

    # make dictionary for results
    results = {}

    # fit each model
    for model_name in models:
        print('Fitting ' + model_name + '...')
        model = models[model_name]
        results[model_name] = []

        # do stream learning
        for i in range(num_pts):
            # compute accuracy every 'eval_num' samples
            if i % eval_num == 0:
                if i != 0:
                    print('Iteration: ', i)
                    preds = model.predict(X_test, mb=512).astype(int)
                    acc = sklearn.metrics.accuracy_score(y_test, preds) * 100
                    results[model_name].append(acc)
            pt = X_train[i, :]
            pt = np.reshape(pt, (1, input_shape))
            label = y_train[i]
            if model_name == 'queue' or model_name == 'reservoir':
                model.fit(pt, label, i)
            else:
                model.fit(pt, label)
        # get final accuracy
        preds = model.predict(X_test, mb=512).astype(int)
        acc = sklearn.metrics.accuracy_score(y_test, preds) * 100
        results[model_name].append(acc)

    # make final plot of all results
    plot_results(results, eval_num, dataset_size)


if __name__ == "__main__":
    main()
