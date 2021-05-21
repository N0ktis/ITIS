import numpy as np


class RecurrentNeuralNetwork:
    def __init__(self, capacity):
        self.k = capacity
        self.__make_network(self.k)
        self.

    def learn(self, study_data):
        pass

    def __make_network(self, capacity):
        for i in enumerate(capacity):
            Perceptron(self.k)


class Perceptron:
    def __init__(self, k):
        self.output = None
        self.w = np.empty(k, dtype=float)

    def get_net(self, output_set, neuron_num):
        net = 0
        for i in range(len(output_set)):
            if i == neuron_num:
                continue
            net += self.w[i] * output_set[i]
        return net

    def activation(self, net):
        if net > 0:
            self.output = 1
        elif net < 0:
            self.output = -1
        else:
            return  # в случае net=0 мы не меняем результат


if __name__ == '__main__':
    study_set = [[-1, -1, 1,  # J
                  -1, -1, 1,
                  -1, -1, 1,
                  1, -1, 1,
                  1, 1, 1],
                 [1, -1, 1,  # K
                  1, -1, 1,
                  1, 1, -1,
                  1, -1, 1,
                  1, -1, 1],
                 [1, -1, -1,  # L
                  1, -1, -1,
                  1, -1, -1,
                  1, -1, -1,
                  1, 1, 1]]
