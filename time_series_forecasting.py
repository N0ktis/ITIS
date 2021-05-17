import numpy as np
import plotly.graph_objs as go


class Prediction_NN:
    def __init__(self, function, left_border, right_border, N=20, epochs=4000, window_width=4, eta=0.9):
        self.function = function
        self.left_border = left_border
        self.right_border = right_border
        self.N = N
        self.epochs = epochs
        self.ETA = eta
        self.window_width = window_width
        self.W = np.array([0. for i in range(self.window_width + 1)])
        self.X = self.__get_X(left_border, right_border, N)
        self.Y = np.array([self.function(x) for x in self.X])

    @staticmethod
    def __get_E(real_func, predicted_func):
        E = 0
        for real, predicted in zip(real_func, predicted_func):
            E += (real - predicted) ** 2
        return np.sqrt(E)

    @staticmethod
    def __get_X(left_border, right_border, N):
        return np.array([x for x in np.arange(left_border, right_border, (right_border - left_border) / N)])

    @staticmethod
    def __get_net(W, X):
        net = W[0]
        for i in range(1, len(W)):
            net += W[i] * X[i - 1]
        return net

    @staticmethod
    def __activation_function(net):
        return net

    def __new_w(self, delta: int, x):
        return self.ETA * delta * x

    def learn(self):
        epoch = 0
        while epoch <= self.epochs:
            for point in range(self.N - self.window_width):
                delta = self.Y[point + self.window_width] - Prediction_NN.__activation_function(
                    Prediction_NN.__get_net(self.W, self.Y[point:point + self.window_width]))
                self.W[0] += self.__new_w(delta, 1)
                for j in range(1, self.window_width + 1):
                    self.W[j] += self.__new_w(delta, self.Y[point + j - 1])
            epoch += 1
        return self.W

    def get_function_prediction(self, left_border, right_border, N):
        new_interval = Prediction_NN.__get_X(left_border, right_border, N)
        real_function = np.empty(self.N + N, float)
        real_function.fill(0.)
        predicted_function = np.empty(self.N + N, float)
        predicted_function.fill(0.)

        for i in range(self.N):
            real_function[i] = self.function(self.X[i])
            predicted_function[i] = self.function(self.X[i])
        for point in range(N):
            real_function[self.N + point] = self.function(new_interval[point])
            predicted_function[self.N + point] = Prediction_NN.__activation_function(
                Prediction_NN.__get_net(self.W, predicted_function[self.N - self.window_width + point:self.N + point]))
        error = Prediction_NN.__get_E(real_function, predicted_function)
        return error, real_function, predicted_function


def graph_builder(interval, F1, F2):
    fig = go.Figure()
    fig.add_trace(go.Scatter({'x': interval, 'y': F1, 'name': 'real'}))
    fig.add_trace(go.Scatter({'x': interval, 'y': F2, 'name': 'prediction'}))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title='График функции sin(t)^2',
                      xaxis_title='t',
                      yaxis_title='x(t)',
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.show()


def get_optimum_M(function, a, b, M_max, eta_max):
    m_arr = [i for i in range(10, M_max + 10, 100)]
    E1 = [None for i in range(0, M_max, 100)]

    for i in range(len(m_arr)):
        A = Prediction_NN(function, a, b, epochs=m_arr[i], window_width=4, eta=eta_max)
        A.learn()
        err, y1, y2 = A.get_function_prediction(b, 2 * b - a, 20)
        E1[i] = err

    fig_M = go.Figure()
    fig_M.add_trace(go.Scatter({'x': m_arr, 'y': E1}))
    fig_M.update_layout(legend_orientation="h",
                        legend=dict(x=.5, xanchor="center"),
                        title='График зависимости E(M)',
                        xaxis_title='M',
                        yaxis_title='E',
                        margin=dict(l=0, r=0, t=30, b=0))
    fig_M.show()


def get_optimum_eta(function, a, b, M_max, eta_max):
    eta_arr = [i / 100 for i in range(1, int(eta_max * 100) + 1, 1)]
    E2 = [None for i in range(1, int(eta_max * 100) + 1, 1)]

    for i in range(len(eta_arr)):
        A = Prediction_NN(function, a, b, epochs=M_max, window_width=4, eta=eta_arr[i])
        A.learn()
        err, y1, y2 = A.get_function_prediction(b, 2 * b - a, 20)
        E2[i] = err

    fig_eta = go.Figure()
    fig_eta.add_trace(go.Scatter({'x': eta_arr, 'y': E2}))
    fig_eta.update_layout(legend_orientation="h",
                          legend=dict(x=.5, xanchor="center"),
                          title='График зависимости E(η)',
                          xaxis_title='η',
                          yaxis_title='E',
                          margin=dict(l=0, r=0, t=30, b=0))
    fig_eta.show()


def function(t):
    return np.sin(t) ** 2


if __name__ == '__main__':
    a = 0
    b = 2
    M = 4000
    P = 4
    Eta = 0.9
    A = Prediction_NN(function, a, b, epochs=4000, window_width=4, eta=0.9)
    print("W=", *A.learn())
    err, y1, y2 = A.get_function_prediction(b, 2 * b - a, 20)
    print("E=", err)
    graph_builder(np.array([x for x in np.arange(a, 2 * b - a, (2 * b - a - a) / 40)]), y1, y2)

    get_optimum_M(function, a, b, M, Eta)
    get_optimum_eta(function, a, b, M, Eta)
