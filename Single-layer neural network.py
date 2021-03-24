from math import exp
import numpy as np
from numba import jit


class Perceptron:
    Et = 0.3

    def __init__(self, N: int, d_activation_func):
        self.N = N
        # self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.W = np.array([0 for i in range(N + 1)])
        self.X = self.__get_X(N)

  #  @jit
    def learn(self, func):
        E = -1
        epoch = 0
        while E != 0:
            curr_F = self.__get_curr_F(self.N, self.W)
            print(curr_F)
            E = self.__get_E(curr_F, func)
            print(E, epoch)

            for i in range(2 ** self.N):
                net = self.__get_net(self.W, self.X[i])
                delta = int(func[i]) - int(curr_F[i])
                self.W[0] = self.__new_w(self.W[0], delta, True, net)
                for j in range(1, self.N + 1):
                    self.W[j] = self.__new_w(self.W[j], delta, self.X[i][j - 1], net)
            epoch += 1
            print(epoch, '%', self.W)

   # @jit
    def __new_w(self, w, delta: int, x: bool, net: int):
        #rint(w + self.Et * delta * x * self.d_activation_func(net), 'h')
        return w + self.Et * delta * x * self.d_activation_func(net)

    @staticmethod
  #  @jit
    def __get_X(N: int):
        X = np.empty((2 ** N, N), bool)
        X.fill(False)
        X[0] = np.array([False, False, False, False])
        for i in range(1, 2 ** N):
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
 #   @jit
    def __get_net(W, X) -> int:
        net = W[0]
        for i in range(1, len(W)):
            net += W[i] * X[i - 1]
            print(W[i],X[i-1])
        print(net,'b')
        return net

    @staticmethod
#    @jit
    def __get_E(S, F) -> int:
        E = 0
        for i in range(len(S)):
            E += (0 if S[i] == F[i] else 1)
        return E

 #   @jit
    def __get_curr_F(self, N: int, W):
        curr_F = np.empty(2 ** (len(W) - 1), bool)
        for i in range(2 ** N):
            curr_F[i] = (True if self.__get_net(W, self.X[i]) >= 0 else False)
        return curr_F


def linear_activation(net):
    return True if net >= 0 else False


def d_linear_activation(net: int) -> int:
    return 1


def sigma_activation(net: int):
    return 1 / (1 + exp(net))


def d_sigma_activation(net: int) -> float:
    return (exp(net)) / (1 + exp(net)) ** 2


F = [False, False, False, True, False, False, True, True, False, False, True, True, False, False, True, True]
# F = [False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, False]
A = Perceptron(4, d_linear_activation)
A.learn(np.array(F))
