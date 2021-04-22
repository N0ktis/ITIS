import copy
from itertools import combinations
from math import exp

import numpy as np
import plotly.graph_objs as go


class Perceptron:
    ETA = 0.3

    def __init__(self, N: int, d_activation_func, bool_func):
        self.N = N
        self.d_activation_func = d_activation_func
        self.bool_func = bool_func
        self.W = np.array([0. for i in range(N + 1)])
        self.X_set = self.get_X_set(N)

    def learn(self, func, X_set, max_epoch=np.inf):
        E_stat = []
        E = -1
        epoch = 0
        while E != 0 or epoch == max_epoch:
            curr_F = self.__get_curr_F(X_set, self.W)
            # print(curr_F)
            E = self.__get_E(curr_F, func)
            E_stat.append(E)
            print(E, epoch)

            for i in range(len(X_set)):
                net = self.__get_net(self.W, X_set[i])
                delta = int(func[i]) - int(curr_F[i])
                self.W[0] += self.__new_w(delta, True, net, i)
                for j in range(1, self.N + 1):
                    self.W[j] += self.__new_w(delta, X_set[i][j - 1], net, i)
            epoch += 1
            # print(epoch, '%', self.W)
        return E, E_stat

    def get_min_vector(self, func, max_epoch=400):
        min_len_comb = 2 ** self.N + 1
        min_comb = None
        min_e_stat = []
        min_w = None
        j = 0
        for i in range(1, 2 ** self.N):
            if i > min_len_comb:
                break
            for X_comb in combinations(self.get_X_set(self.N), i):
                j += 1
                self.W.fill(0.)
                min_func = np.empty(len(X_comb), bool)
                min_func.fill(False)
                for k in range(i):
                    min_func[k] = self.bool_func(X_comb[k])
                E, e_stat = self.learn(min_func, X_comb, max_epoch)
                if self.__get_E(self.__get_curr_F(self.X_set, self.W), func) == 0 and len(X_comb) < min_len_comb:
                    min_len_comb = len(X_comb)
                    min_comb = X_comb
                    min_w = copy.copy(self.W)
                    print(min_comb, min_w, self.W, min_len_comb)
                    min_e_stat = e_stat
        print(min_comb, min_w, self.W)

    def __new_w(self, delta: int, x: bool, net: int, i):
        return self.ETA * delta * x * self.d_activation_func(net)

    @staticmethod
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

    @staticmethod
    def __get_net(W, X) -> int:
        net = W[0]
        for i in range(1, len(W)):
            net += W[i] * X[i - 1]
        return net

    @staticmethod
    def __get_E(S, F) -> int:
        E = 0
        for i in range(len(F)):
            E += (0 if S[i] == F[i] else 1)
        return E

    @staticmethod
    def __get_curr_F(X_set, W):
        curr_F = np.empty(len(X_set), bool)
        for i in range(len(X_set)):
            curr_F[i] = (True if Perceptron.__get_net(W, X_set[i]) >= 0 else False)
        return curr_F


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


def linear_activation(net) -> int:
    return 1 if net >= 0 else 0


def d_linear_activation(net: int) -> int:
    return 1


def sigma_activation(net: int):
    return 1 / (1 + exp(net))


def d_sigma_activation(net: int) -> float:
    return (exp(net)) / (1 + exp(-net)) ** 2  # я сразу упростил производную ФА, чтобы не делать лишних вызовов


def get_bool_func(X_vector):
    return (X_vector[0] + X_vector[1] + X_vector[3]) * X_vector[2]


if __name__ == '__main__':
    F = [False, False, False, True, False, False, True, True, False, False, True, True, False, False, True, True]
    A = Perceptron(4, d_linear_activation, get_bool_func)
    A.learn(np.array(F), A.get_X_set(4))
    B = Perceptron(4, d_sigma_activation, get_bool_func)
    B.learn(np.array(F), B.get_X_set(4))
    A.get_min_vector(np.array(F))
    B.get_min_vector(np.array(F))
