import os
import math
import matplotlib.pyplot as plt


def main():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    fig, ax = plt.subplots()
    x = [i for i in range(1, 100)]
    y = [math.pow(i, -1) for i in x]
    plt.plot(x, y, label='x^-1')

    y = [math.pow(i, -2) for i in x]
    plt.plot(x, y, label='x^-2')

    y = [math.pow(i, -3) for i in x]
    plt.plot(x, y, label='x^-3')

    y = [math.pow(10, 1/i) for i in x]
    plt.plot(x, y, label='10^1/x')

    y = [math.pow(2, 1 / i) for i in x]
    plt.plot(x, y, label='2^1/x')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()