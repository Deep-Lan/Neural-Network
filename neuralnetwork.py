import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NeuralNetwork(object):
    def __init__(self, cell_num_list):
        """
        :param cell_num_list: a list,each element is cell number of every layer.
        For example,[4, 10, 3] means 3 layers,4 cells in input layer,10 cells in hidden layer and 3 cells in output layer
        """
        self.__layer_num = len(cell_num_list) - 1
        self.__weight_list = []
        self.__bias_list = []
        for i in range(self.__layer_num):
            weight = np.random.normal(size=(cell_num_list[i], cell_num_list[i + 1]))
            bias = np.zeros((1, cell_num_list[i + 1]))
            self.__weight_list.append(weight)
            self.__bias_list.append(bias)

    def predict(self, x):
        """
        Do rediction by net.
        :param x: input,can be one sample or multiple sample
        :return: a,the prediction output of network on data x
        """
        if len(x.shape) == 1:
            x.reshape(-1, len(x))
        a = x
        for i in range(self.__layer_num):
            z = np.dot(a, self.__weight_list[i]) + self.__bias_list[i]
            a = sigmoid(z)
        return a

    def __CostGradient(self, x, y):
        """
        Compute the cost and gradient using input and label.
        :param x: input
        :param y: label
        :return: cost and a list of gradient of cost function in each layer
        """
        a_list = []
        a = x
        for i in range(self.__layer_num):
            a_list.append(a)
            z = np.dot(a, self.__weight_list[i]) + self.__bias_list[i]
            a = sigmoid(z)
        cost = np.sum((y - a) ** 2) / 2 / len(a)
        delta_list = []
        delta = (a - y) * a * (1 - a)
        for i in range(self.__layer_num):
            delta_list.append(delta)
            delta = np.dot(delta, self.__weight_list[-1 - i].T) * a_list[-1 - i] * (1 - a_list[-1 - i])
        delta_list.reverse()
        bias_gradient_list = []
        for i in range(self.__layer_num):
            bias_gradient = np.sum(delta_list[i], axis=0, keepdims=True) / len(x)
            bias_gradient_list.append(bias_gradient)
        weight_gradient_list = []
        for i in range(self.__layer_num):
            weight_gradient = np.dot(a_list[i].T, delta_list[i]) / len(x)
            weight_gradient_list.append(weight_gradient)
        return cost, weight_gradient_list, bias_gradient_list

    def train(self, x, y, alpha=0.1, iteration_num=10000):
        """
        Train the neural network.
        :param x: inputs of train dataset
        :param y: labels of train dataset
        :param alpha: learning rate
        :param iteration_num: the maximum iteration number
        """
        cost_list = []
        for _ in range(iteration_num):
            cost, weight_gradient_list, bias_gradient_list = self.__CostGradient(x, y)
            cost_list.append(cost)
            for i in range(self.__layer_num):
                self.__weight_list[i] -= alpha * weight_gradient_list[i]
                self.__bias_list[i] -= alpha * bias_gradient_list[i]
        plt.plot(range(iteration_num), cost_list), plt.xlabel('iteration number'), plt.ylabel('cost')


def ComputeAccuracy(net, x_test, y_test):
    """
    Compute the accuracy of net
    :param net: the neural network
    :param x_test: inputs of test dataset
    :param y_test: labels of test datset
    :return: the accuracy in test datset
    """
    pred = net.predict(x_test)
    temp = 0
    row, column = pred.shape
    for i in range(row):
        for j in range(column):
            if pred[i, j] >= 0.5:
                pred[i, j] = 1
            else:
                pred[i, j] = 0
        if (pred[i] == y_test[i]).all():
            temp += 1
    acc = temp / row
    return acc


def ReadData(filename):
    """
    Read dataset from "filename".
    :param filename: a string of a file that contains dataset
    :return:x_train, y_train, x_test, y_test 4 datasets.
            In the each of the dataset,a row means a sample and a column means a dimension.
    """
    f = open(filename, 'r')
    dataset = f.readlines()
    f.close()
    for i in range(len(dataset)):
        dataset[i] = dataset[i].rstrip().split(',')
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
    random.shuffle(dataset)
    test_dataset = dataset[:50]
    train_dataset = dataset[50:]
    x_train = []
    y_train = []
    for data in train_dataset:
        x_train.append(data[:-1])
        if data[-1] == 'Iris-setosa':
            y_train.append([1, 0, 0])
        if data[-1] == 'Iris-versicolor':
            y_train.append([0, 1, 0])
        if data[-1] == 'Iris-virginica':
            y_train.append([0, 0, 1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = []
    y_test = []
    for data in test_dataset:
        x_test.append(data[:-1])
        if data[-1] == 'Iris-setosa':
            y_test.append([1, 0, 0])
        if data[-1] == 'Iris-versicolor':
            y_test.append([0, 1, 0])
        if data[-1] == 'Iris-virginica':
            y_test.append([0, 0, 1])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def main1():
    """
    this function is to realize the stage1 of experiment
    """
    # read data
    x_train, y_train, x_test, y_test = ReadData('data.txt')

    # generate a neural network
    net = NeuralNetwork([4, 20, 10, 3])

    # train the neural network
    net.train(x_train, y_train)

    # compute the accuracy of net
    acc = ComputeAccuracy(net, x_test, y_test)
    print('Accuracy is:', acc)

    # show image
    plt.show()


def main2():
    """
    this function is to realize the stage2 of experiment,which is for autoencoder
    """
    # negerate a set of data
    x = np.eye(10)
    x = np.concatenate((x, x, x, x, x, x, x, x, x, x), axis=0)

    # generate a autoencoder neural network,cells in input and output layers are more,but cells in hidden layer are less
    # you can change cell number of hidden layer to any integer value less than 10
    autoencoder_net = NeuralNetwork([10, 5, 10])

    # train the neural network
    autoencoder_net.train(x, x)

    # compute the accuracy of net
    acc = ComputeAccuracy(autoencoder_net, x, x)
    print('Accuracy is:', acc)

    # show image
    plt.show()


if __name__ == '__main__':
    main1()
    # main2()