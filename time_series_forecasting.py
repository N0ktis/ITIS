import numpy as np


class Perceptron:
    def __init__(self, x_list, y_list, et=1, window_width=4):
        self.x_list = x_list
        self.y_list = y_list
        self.et = et
        self.window_width = window_width
        self.W = np.array([0. for i in range(self.window_width + 1)])

    def learn(self):
        pass


def function(t):
    return np.sin(t) ** 2


class Perceptron:

    def __init__(self, p, M, ny, Y1_real, X1):
        self.p = p
        self.M = M
        self.ny = ny
        self.synaptic_weights = [0 for it in range(p + 1)]
        self.Y1_real = Y1_real
        self.X1 = X1

    def neuron(self, values):
        return self.synaptic_weights[0] + sum(self.synaptic_weights[k + 1] * values[k] for k in range(self.p))

    def correction(self, values, delta):
        self.synaptic_weights[0] += self.ny * delta
        for i in range(len(self.synaptic_weights) - 1):
            self.synaptic_weights[i + 1] += self.ny * values[i] * delta

    def learning(self):
        for epoch in range(self.M):
            y = []
            for i in range(self.p):
                y.append(self.Y1_real[i])

            for t in self.Y1_real[self.p:]:
                values = y[:-self.p - 1:-1]
                values.reverse()
                yi = self.neuron(values)
                self.correction(values, t - yi)
                y.append(yi)
        return y

    def window(self, N, Yp):
        y = Yp
        for step in range(N):
            values = y[:-self.p - 1:-1]
            values.reverse()
            yi = self.neuron(values)
            y.append(yi)
        return y[self.p + 1:]
# 0.13113028064759905 3000 7 0.1


import math

import matplotlib.pyplot as plt

from perceptron import Perceptron


def function(t):
    return (math.sin(t))**2
    #return math.sin(0.1 * (t ** 3) - 0.2 * (t ** 2) + t - 1)


def error(Y, Y_real):
    sum = 0
    for i in range(len(Y_real)):
        sum += pow((Y_real[i] - Y[i]), 2)
    return math.sqrt(sum)


def Plot(X, Y, Y_real, b):
    fig, ax = plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X, Y, X, Y_real)
    ax.grid()
    ax.vlines(b, min(Y), max(Y), color='r')

    plt.show()


def research(X, Y, Y2, N):
    E = []
    M = []
    for m in range(500, 10500, 500):
        perceptron = Perceptron(13, m, 0.1, Y, X)
        perceptron.learning()
        Y_p = perceptron.window(N, Y1_real[:- 14:-1])
        M.append(m)
        E.append(error(Y2, Y_p))
    plt.xlabel("M")
    plt.ylabel("E")
    plt.plot(M, E)
    plt.show()
    E = []
    P = []
    for p in range(2, 20):
        perceptron = Perceptron(p, 3500, 0.1, Y, X)
        perceptron.learning()
        Y_p = perceptron.window(N, Y1_real[:- p - 1:-1])
        P.append(p)
        E.append(error(Y2, Y_p))
    plt.xlabel("P")
    plt.ylabel("E")
    plt.plot(P, E)
    plt.show()

    E = []
    NY = []
    for ny in range(10, 105, 5):
        perceptron = Perceptron(13, 3500, ny / 100, Y, X)
        perceptron.learning()
        Y_p = perceptron.window(N, Y1_real[:- 14:-1])
        NY.append(ny / 100)
        E.append(error(Y2, Y_p))
    plt.xlabel("NY")
    plt.ylabel("E")
    plt.plot(NY, E)
    plt.show()


def brute_force(X1, Y1_real, Y2_real, N):
    e = 10
    for M in range(500, 10500, 500):
        for p in range(2, 17):
            for ny in range(10, 105, 5):
                print("M=", M, "p=", p, "ny=", ny / 100)
                perceptron = Perceptron(p, M, ny / 100, Y1_real, X1)
                Y1 = perceptron.learning()
                Y2 = perceptron.window(N, Y1_real[:- p - 1:-1])
                new_e = error(Y2, Y2_real)
                print(new_e)
                if new_e < e:
                    top_perceptron = perceptron
                    e = new_e

    print(e, top_perceptron.M, top_perceptron.p, top_perceptron.ny)
    # 2.868558139153208 1500 3 0.55
    # 1.196485296498305 3500 13 0.1


a = 0
b = 2
N = 20

p = 7
M = 3000
ny = 0.1

X1 = []
Y1_real = []
i = a
while i <= b:
    X1.append(i)
    Y1_real.append(function(i))
    i += (b - a) / N

X2 = []
Y2_real = []
i = b + (2 * b - a - b) / N
while i <= (2 * b - a):
    X2.append(i)
    Y2_real.append(function(i))
    i += (2 * b - a - b) / N

#brute_force(X1, Y1_real, Y2_real, N)

research(X1, Y1_real, Y2_real, N)

#perceptron = Perceptron(p, M, ny, Y1_real, X1)
#Y1 = perceptron.learning()
#Y2 = perceptron.window(N, Y1_real[:- p - 1:-1])
#print("e = ", error(Y2, Y2_real), "synaptic weights:", perceptron.synaptic_weights)
#Plot(X1 + X2, Y1 + Y2, Y1_real + Y2_real, b)
