from math import exp
import numpy as np


class Perceptron:
    Et = 0.3

    def __init__(self, N: int, activation_func, d_activation_func):
        self.N = N
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.W = np.array([0 for i in range(np.sqrt(N) + 1)])

    @staticmethod
    def get_net(w, l):
        # print(w,'@',l)
        net = w[0]
        for i in range(1, len(w)):
            # net += w[i] * (1 if X[l][i] is True else 0)
            net += w[i] * X[l][i - 1]
        return net

    @staticmethod
    def get_E(S, F):
        E = 0
        for i in range(len(S)):
            E += (0 if S[i] == F[i] else 1)
        return E

    @staticmethod
    def get_new_S(W):
        new_F = []
        for l in range(len(X)):
            new_F.append(True if get_net(W, l) >= 0 else False)
        return new_F

    def learn(self, func):
        E = 13
        k = 0
        while E != 0:
            S = get_new_S(W)
            print(S)
            E = get_E(S, F)
            print(E, k)

            for i in range(len(S)):
                net = get_net(W, i)
                delta = F[i] - S[i]
                W[0] = W[0] + new_w(W[0], delta, True) * 1
                for j in range(1, len(W)):
                    W[j] = W[j] + new_w(W[j], delta, X[i][j - 1]) * 1
            k += 1
            print(k, '%', W)


def linear_activation(W, l):
    return True if get_net(W, l) >= 0 else False


def sigma_activation(W, l):
    return 1 / (1 + exp(-get_net(W, l)))


Et = 0.3
X = [[False, False, False, False],
     [False, False, False, True],
     [False, False, True, False],
     [False, False, True, True],
     [False, True, False, False],
     [False, True, False, True],
     [False, True, True, False],
     [False, True, True, True],
     [True, False, False, False],
     [True, False, False, True],
     [True, False, True, False],
     [True, False, True, True],
     [True, True, False, False],
     [True, True, False, True],
     [True, True, True, False],
     [True, True, True, True]]

"""
def new_w(w, d, x):
    return w + Et * d * x
"""


def new_w(w, d, x):
    return Et * d * x


def get_net(w, l):
    # print(w,'@',l)
    net = w[0]
    for i in range(1, len(w)):
        # net += w[i] * (1 if X[l][i] is True else 0)
        net += w[i] * X[l][i - 1]
    return net


def get_E(S, F):
    E = 0
    for i in range(len(S)):
        E += (0 if S[i] == F[i] else 1)
    return E


def get_new_S(W):
    new_F = []
    for l in range(len(X)):
        new_F.append(True if get_net(W, l) >= 0 else False)
    return new_F


def linear_activation(W, l):
    return True if get_net(W, l) >= 0 else False


def sigma_activation(W, l):
    return 1 / (1 + exp(-get_net(W, l)))


W = [0, 0, 0, 0, 0]
S = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
F = [False, False, False, True, False, False, True, True, False, False, True, True, False, False, True, True]
# F = [False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, False]

E = 13
k = 0
while E != 0:
    S = get_new_S(W)
    print(S)
    E = get_E(S, F)
    print(E, k)

    for i in range(len(S)):
        net = get_net(W, i)
        delta = F[i] - S[i]
        W[0] = W[0] + new_w(W[0], delta, True) * 1
        for j in range(1, len(W)):
            W[j] = W[j] + new_w(W[j], delta, X[i][j - 1]) * 1
    k += 1

print(k, '%', W)
W = [0, 0, 0, 0, 0]
S = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
E = 13
k = 0
while E != 0:
    S = get_new_S(W)
    print(S)
    E = get_E(S, F)
    print(E, k)

    for i in range(len(S)):
        net = get_net(W, i)
        delta = F[i] - S[i]
        W[0] = W[0] + new_w(W[0], delta, True) * sigma_activation(W, i) * (1 - sigma_activation(W, i))
        for j in range(1, len(W)):
            W[j] = W[j] + new_w(W[j], delta, X[i][j - 1]) * sigma_activation(W, i) * (1 - sigma_activation(W, i))
    k += 1

print(k, W)
