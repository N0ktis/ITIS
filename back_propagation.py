import numpy as np
import plotly.graph_objs as go


class Perceptron:
    ETA = 1

    def __init__(self, cnt_inputs: int, activation_func, d_activation_func=None, hidden=False):
        self.__inputs = cnt_inputs
        self.__activation_func = activation_func
        self.__d_activation_func = d_activation_func
        self.__is_hidden = hidden
        start_weight = 0.
        if hidden:
            start_weight = 0.5
        self.W = np.array([start_weight for i in range(cnt_inputs + 1)])

    def __new_w(self, delta: float, x: float):
        return Perceptron.ETA * delta * x

    @staticmethod
    def __get_net(W, input_set) -> int:
        net = W[0]
        for i in range(1, len(W)):
            net += W[i] * input_set[i - 1]
        return net

    def get_delta(self, elder_val, curr_val, weights=None):
        delta = 0
        if self.__is_hidden:
            for value, weight in zip(elder_val, weights):
                delta += value * weight
            return self.__d_activation_func(curr_val) * delta
        else:
            return self.__d_activation_func(curr_val) * (elder_val - curr_val)

    def activate(self, input_set):
        return self.__activation_func(Perceptron.__get_net(self.W, input_set))

    def set_weights(self, delta, input_set):
        self.W[0] += self.__new_w(delta, 1)
        print(self.__inputs, delta, input_set,self.W)
        for index in range(1, self.__inputs + 1):
            print(index)
            self.W[index] += self.__new_w(delta, input_set[index - 1])


class NeuralNetwork:
    def __init__(self, model):
        # self.cnt_inputs = model[0]
        # self.input_data = input_data
        # self.cnt_outputs = model[len(model) - 1]
        # self.output_data = output_data
        # self.layers = model[1:len(model) - 2]
        model[0] = model[1] // model[0]
        self.model = model
        self.__layer_cnt = len(model) - 1
        self.__architecture = np.empty(self.__layer_cnt, object)
        self.__out_in_data = np.empty(self.__layer_cnt + 1, object)
        self.__delta = np.empty(self.__layer_cnt, object)

    @staticmethod
    def __get_E(current_result, real_result):
        sqr_sum = 0
        for curr_value, real_value in zip(current_result, real_result):
            sqr_sum += (curr_value - real_value) ** 2
        return np.sqrt(sqr_sum)

    def build_nn(self, input_data, activation_func, d_activation_func):
        self.__out_in_data[0] = input_data
        hidden = True
        for layer in range(self.__layer_cnt):
            if layer == self.__layer_cnt - 1:
                hidden = False
            self.__architecture[layer] = np.empty(self.model[layer + 1], object)
            self.__out_in_data[layer + 1] = np.empty(self.model[layer + 1], float)
            self.__delta[layer] = np.empty(self.model[layer + 1], float)
            for neuron in range(len(self.__architecture[layer])):
                self.__architecture[layer][neuron] = Perceptron(self.model[layer], activation_func, d_activation_func,
                                                                hidden)
                self.__out_in_data[layer + 1][neuron] = self.__architecture[layer][neuron].activate(
                    self.__out_in_data[layer])
        print(self.__architecture)

    def learning(self, output_data):
        E = self.__get_E(self.__out_in_data[self.__layer_cnt], output_data)
        epochs = 0
        # while E >= 0.0001:
        print(self.__delta, self.__delta[0][0])
        for neuron in range(len(self.__architecture[self.__layer_cnt - 1])):
            self.__delta[self.__layer_cnt - 1][neuron] = self.__architecture[self.__layer_cnt - 1][
                neuron].get_delta(output_data[neuron], self.__out_in_data[self.__layer_cnt][neuron])
        for layer in range(self.__layer_cnt - 2, -1, -1):
            for neuron in range(len(self.__architecture[layer])):
                self.__delta[layer][neuron] = self.__architecture[layer][neuron].get_delta(self.__delta[layer + 1],
                                                                                           self.__out_in_data[
                                                                                               layer + 1][neuron],
                                                                                           np.array(
                                                                                               [elder_neurons.W[
                                                                                                    neuron + 1] for
                                                                                                elder_neurons in
                                                                                                self.__architecture[
                                                                                                    layer + 1]]))
        for layer in range(self.__layer_cnt):
            for neuron in range(len(self.__architecture[layer])):
                self.__architecture[layer][neuron].set_weights(self.__delta[layer][neuron],
                                                               self.__out_in_data[layer][neuron])
                print(self.__architecture[layer][neuron].W)
        print(self.__delta)


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


def activation_function(net: int) -> bool:
    return (1 - np.exp(-net)) / (1 + np.exp(-net))


def d_activation_function(f_net: float) -> float:
    return 0.5 * (1 - f_net ** 2)  # я сразу упростил производную ФА, чтобы не делать лишних вызовов


if __name__ == '__main__':
    # N = 4
    # F = [False, False, False, True, False, False, True, True, False, False, True, True, False, False, True, True]
    # learning(N, F, linear_activation, d_linear_activation, True)
    # learning(N, F, sigma_activation, d_sigma_activation, True)
    A = NeuralNetwork([3, 3, 4])
    A.build_nn(np.array([0.3, -0.1, 0.9]), activation_function, d_activation_function)
    A.learning(np.array([0.1, -0.6, 0.2, 0.7]))
