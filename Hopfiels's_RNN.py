import numpy as np


class RecurrentNeuralNetwork:
    def __init__(self, capacity):
        self.k = capacity


class Perceptron:
    def __init__(self):
        self.output = None
        self.w = None

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
        return self.output


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
