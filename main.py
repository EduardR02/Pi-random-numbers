
from numba import vectorize
import numpy as np
import time


@vectorize(['float32(float32, float32)'], target='cuda')
def my_calc(a, b):
    return ((a * 2 - 1) ** 2) + ((b * 2 - 1) ** 2)


def my_calc_cpu(a, b, c):
    # this is intentionally not using numpy efficiency to show the difference between computing on cpu and gpu
    for i in range(a.size):
        c[i] = ((a[i] * 2 - 1) ** 2) + ((b[i] * 2 - 1) ** 2)


def compute_pi():
    num = 100000
    start = time.time()
    a = np.array(np.random.random(num), dtype=np.float32)
    b = np.array(np.random.random(num), dtype=np.float32)
    c = np.zeros(num, dtype=np.float32)
    print("Numpy init time:", time.time() - start)
    start = time.time()
    my_calc_cpu(a, b, c)
    print("Cpu computing time:", time.time() - start)
    count = np.sum(c <= 1)
    if count == 0: print("Count is 0")
    else: print("Pi is:", (count / num) * 4)


def main_compute_pi():
    num = 200000000
    repeat = 10
    x = 0
    for i in range(repeat):
        start = time.time()
        a = np.array(np.random.random(num), dtype=np.float32)
        b = np.array(np.random.random(num), dtype=np.float32)
        print("Numpy init time:", time.time() - start)
        start = time.time()
        c = my_calc(a, b)
        print("Gpu computing time:", time.time() - start)
        x += np.sum(c <= 1)
    if not x: print("Count is 0")
    else: print("Pi is:", (x / (num * repeat)) * 4)


if __name__ == '__main__':
    compute_pi()
    main_compute_pi()
