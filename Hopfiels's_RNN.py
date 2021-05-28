import numpy as np


class RecurrentNeuralNetwork:
    def __init__(self, capacity):
        self.k = capacity
        self.layer = np.empty(capacity, dtype=object)
        self.__fill_layer(self.k)
        self.output = np.empty(capacity, dtype=float)

    def learn(self, study_data):
        for weight_num in range(self.k):
            for perceptron_num in range(self.k):
                if weight_num == perceptron_num:
                    self.layer[perceptron_num].w[weight_num] = 0
                    continue
                init_value = 0
                for value in study_data:
                    init_value += value[weight_num] * value[perceptron_num]
                self.layer[perceptron_num].w[weight_num] = init_value

        print("Weight matrix:")
        for perceptron in self.layer:
            print(perceptron.w)

    def restoring(self, image):
        flag = True
        for i in range(len(image)):
            self.layer[i].output = image[i]
            self.output[i] = image[i]
        while flag:
            flag = False
            for neuron in range(self.k):
                current_neuron = self.layer[neuron]
                current_neuron.activation(current_neuron.get_net(self.output))
                if self.output[neuron] != current_neuron.output:
                    flag = True
                    self.output[neuron] = current_neuron.output

    def __fill_layer(self, capacity):
        for i in range(capacity):
            self.layer[i] = Perceptron(self.k)


class Perceptron:
    def __init__(self, k):
        self.output = None
        self.w = np.empty(k, dtype=float)

    def get_net(self, output_set):
        net = 0
        for i in range(len(output_set)):
            net += self.w[i] * output_set[i]
        return net

    def activation(self, net):
        if net > 0:
            self.output = 1
            return True
        elif net < 0:
            self.output = -1
            return True
        else:
            return False


def get_image(px_data_array, height, weight):
    image_line = []

    for h in range(height):
        for px in px_data_array[h::height]:
            if px == 1:
                image_line.append('■')
            else:
                image_line.append('□')
        print(*image_line)
        image_line.clear()
    print()


if __name__ == '__main__':
    study_set = [[-1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1],  # J
                 [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1],  # K
                 [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1]]  # L

    J_false = [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1]
    K_false = [1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1]
    L_false = [1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1]
    A = RecurrentNeuralNetwork(15)
    A.learn(study_set)

    print('Spoiled "J"')
    get_image(J_false, 5, 3)
    print('Restored "J"')
    A.restoring(J_false)
    get_image(A.output, 5, 3)
    print('Spoiled "K"')
    get_image(K_false, 5, 3)
    print('Restored "K"')
    A.restoring(K_false)
    get_image(A.output, 5, 3)
    print('Spoiled "L"')
    get_image(L_false, 5, 3)
    print('Restored "L"')
    A.restoring(L_false)
    get_image(A.output, 5, 3)
