from math import exp

import numpy as np
import plotly.graph_objs as go


class Perceptron:
    ETA = 1

    def __init__(self, cnt_inputs: int, activation_func, d_activation_func=None, real_value=None, hidden=False):
        self.__inputs = cnt_inputs
        self.__activation_func = activation_func
        self.__d_activation_func = d_activation_func
        self.__is_hidden = hidden
        self.real_value = real_value
        self.W = np.array([0. for i in range(cnt_inputs + 1)])

    def __new_w(self, delta: int, x: bool, net: int):
        return self.ETA * delta * x * self.__d_activation_func(net)

    @staticmethod
    def __get_net(W, input_set) -> int:
        net = W[0]
        for i in range(1, len(W)):
            net += W[i] * input_set[i - 1]
        return net

    def activate(self, input_set):
        if self.__is_hidden:
            return self.__activation_func(self.real_value, input_set)
        else:
            return self.__activation_func(Perceptron.__get_net(self.W, input_set))

    def set_weights(self, result, input_set):
        delta = self.real_value - result
        net = Perceptron.__get_net(self.W, input_set)
        self.W[0] += self.__new_w(delta, True, net)
        for index in range(1, self.__inputs + 1):
            self.W[index] += self.__new_w(delta, input_set[index - 1], net)


def graph_builder(E_stat):
    fig = go.Figure()
    fig.add_trace(go.Scatter({'x': [i for i in range(len(E_stat))], 'y': E_stat, 'name': 'E'}))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title='График суммарной ошибки НС по эпохам обучения',
                      xaxis_title='Эпохи',
                      yaxis_title='Суммарная ошибка',
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.show()


def get_X_set(N: int):
    X = np.empty((2 ** N, N), bool)
    X.fill(False)
    for i in range(2 ** N):
        pos = N - 1
        buf_i = i
        while buf_i != 0:
            if (buf_i % 2) == 1:
                X[i][pos] = True
            else:
                X[i][pos] = False
            pos -= 1
            buf_i = buf_i // 2
    return X


def get_cnt_hidden_neurons(bool_func):
    true_index_set = set()
    false_index_set = set()
    for index in range(len(bool_func)):
        if bool_func[index]:
            true_index_set.add(index)
        else:
            false_index_set.add(index)
    if len(true_index_set) <= len(false_index_set):
        return true_index_set
    else:
        return false_index_set


def gauss_activation(current, center):
    sqr_sum = 0
    for curr_value, center_value in zip(current, center):
        sqr_sum += ((1 if curr_value else 0) - (1 if center_value else 0)) ** 2
    return np.exp(-sqr_sum)


def linear_activation(net:) -> bool:
    return True if net >= 0 else False


def d_linear_activation(net: int) -> int:
    return 1


def sigma_activation(net: int) -> bool:
    return True if (1 / (1 + exp(-net))) >= 0.5 else False


def d_sigma_activation(net: int) -> float:
    return (exp(net)) / (1 + exp(net)) ** 2  # я сразу упростил производную ФА, чтобы не делать лишних вызовов


def get_bool_func(X_vector):
    return (X_vector[0] + X_vector[1] + X_vector[3]) * X_vector[2]


def main(N, bool_func):
    epochs = -1
    E = -1
    E_stat = []
    rbf_neurons_center = get_cnt_hidden_neurons(bool_func)
    all_x_set = get_X_set(N)
    rbf_neurons_layer = np.empty(len(rbf_neurons_center), object)
    size_rbf_neurons_layer = len(rbf_neurons_layer)
    i = 0
    print(rbf_neurons_layer)
    # main_neuron = Perceptron(size_rbf_neurons_layer, linear_activation, d_linear_activation)
    main_neuron = Perceptron(size_rbf_neurons_layer, sigma_activation, d_sigma_activation)
    for index in rbf_neurons_center:
        rbf_neurons_layer[i] = Perceptron(N, gauss_activation, real_value=all_x_set[index], hidden=True)
        i += 1
    while E != 0:
        E = 0
        for x_set, value in zip(all_x_set, bool_func):
            main_neuron.real_value = value
            fi_set = np.empty(size_rbf_neurons_layer, float)
            for index in range(size_rbf_neurons_layer):
                fi_set[index] = rbf_neurons_layer[index].activate(x_set)
            # print(fi_set,'!')
            result = main_neuron.activate(fi_set)
            main_neuron.set_weights(result, fi_set)
            print(result, value)
            E += (0 if value == result else 1)
        epochs += 1
        print(epochs, main_neuron.W, E)
        E_stat.append(E)
    graph_builder(E_stat)


if __name__ == '__main__':
    F = [False, False, False, True, False, False, True, True, False, False, True, True, False, False, True, True]
    main(4, F)
