from math import exp
import numpy as np
import itertools

print(list(itertools.combinations_with_replacement([True, False], 4)))
print(list(itertools.combinations_with_replacement([False, True], 4)))
a = True - False
b = True
print(b * 0.5)
print(a)

print(exp(2))
print([0 for i in range(3)])


def binary(a):
    s = ''
    while a != 0:
        s += str (a % 2)
        a = a // 2
    print(s)

binary(1)
binary(0)
binary(2)
binary(7)


def get_X(N: int):
    X = np.empty((2 ** N, N), bool)
    for i in range(2 ** N):
        pos = N - 1
        ii=i
        while ii != 0:
            print(i)
            if (ii % 2) == 1:
                X[i][pos] = True
            else:
                X[i][pos] = False
            pos -= 1
            ii = ii // 2
    return X

print(get_X(4))



print(False-True)