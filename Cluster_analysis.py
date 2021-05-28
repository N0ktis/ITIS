import json

import numpy as np
import prettytable


class Neuron:
    def __init__(self, center_value):
        self.center = center_value
        self.data = []

    def activation(self, value):
        return np.sqrt((value - self.center) ** 2)


class KohonenNeuralNetwork:
    def __init__(self, cluster_radius):
        self.radius = cluster_radius
        self.koh_layer = []

    def get_clusters(self, data, cluster_feature):
        if len(data) == 0:
            return
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
    table = prettytable.PrettyTable()
    for cluster in NN.koh_layer:
        print("\nCluster's center:", cluster.center, "Cluster's size:", len(cluster.data))
        table.field_names = ["ID", "Address", "Car capacity"]
        for value in cluster.data:
            table.add_row([value['ID'], value['Address'], value['CarCapacity']])
        print(table)
        table.clear()


def get_data(file_path, sample_size):
    json_file = open(file_path)
    data = json.load(json_file)
    return data[:sample_size]


if __name__ == '__main__':
    data = get_data('data-4905-2021-03-09.json', 200)
    radius = 5
    A = KohonenNeuralNetwork(radius)
    A.get_clusters(data, 'CarCapacity')
    print_clusters(A)
