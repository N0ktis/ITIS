import json

import numpy as np


def binary(a):
    s = ''
    while a != 0:
        s += str(a % 2)
        a = a // 2
    # print(s)


# binary(1)
# binary(0)
# binary(2)
# binary(7)


def get_X(N: int):
    X = np.empty((2 ** N, N), bool)
    for i in range(2 ** N):
        pos = N - 1
        ii = i
        while ii != 0:
            print(i)
            if (ii % 2) == 1:
                X[i][pos] = True
            else:
                X[i][pos] = False
            pos -= 1
            ii = ii // 2
    return X


def get_bool_func(X_vector):
    return (X_vector[0] + X_vector[1] + X_vector[3]) * X_vector[2]


js = open('data-4905-2021-03-09.json')
rjs = json.load(js)
length = len(rjs)
summa = 0
maxxx = -1
minnn = 90000000
"""for value in rjs:
    if value['CarCapacity'] >= maxxx:
        maxxx = value['CarCapacity']
    if value['CarCapacity'] < minnn:
        minnn = value['CarCapacity']
    summa += value['CarCapacity']
print(rjs[0]['CarCapacity'])
print(summa / length, maxxx,minnn)"""

for i in range(20):
    print(rjs[i]['CarCapacity'], rjs[i])
