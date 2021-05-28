import json

import numpy as np


class Neuron:
    def __init__(self, center_value):
        self.center = center_value
        self.data = []

    def activation(self, value):
        return np.sqrt((value - self.center) ** 2)


class KohonenNeuralNetwork:
    def __init__(self, cluster_radius):
        self.radius = cluster_radius
        self.koh_layer = []  # np.empty(capacity, dtype=object)

    def get_clusters(self, data, cluster_feature):
        if len(self.koh_layer) == 0:
            self.koh_layer.append(Neuron(data[0][cluster_feature]))

        for value in data:
            min_r = np.inf
            cluster = None
            for neuron in self.koh_layer:
                value_radius = neuron.activation(value[cluster_feature])
                if value_radius < min_r:
                    min_r = value_radius
                    cluster = neuron

            if min_r > self.radius:
                new_neuron = Neuron(value[cluster_feature])
                new_neuron.data.append(value)
                self.koh_layer.append(new_neuron)
            else:
                cluster.data.append(value)


def print_clusters(NN):
    pass


def get_data(file_path):
    pass


if __name__ == '__main__':
    js = open('data-4905-2021-03-09.json')
    rjs = json.load(js)
    A = KohonenNeuralNetwork(0)
    A.get_clusters(rjs[:20], 'CarCapacity')
    for i in A.koh_layer:
        print(i.center)
        for j in i.data:
            print(j['ID'], j['CarCapacity'])
