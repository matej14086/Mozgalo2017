import numpy as np
from PIL import Image
import os,sys

def intersection(method1="cnn.npy",method2="labels_cnn.npy",method3="labels_hog.npy"):
    a = np.load(method2)
    b = np.load(method1)
    c = np.load(method3)
    kon = np.zeros((10, 4), dtype='int32')

    for i in range(10):
        max = 0
        kon[i, 1] = i
        for j in range(10):
            for k in range(10):
                m = (b == i)
                n = (a == j)
                l = (c == k)
                h = (m * n)
                h = (h * l)
                if sum(h) > max:
                    max = sum(h)
                    kon[i, 0] = j
                    kon[i, 2] = k
                    kon[i, 3] = max
    return kon

def raz(source, target,method1="cnn.npy",method2="labels_cnn.npy",method3="labels_hog.npy"):
    root = os.getcwd() + '\\'
    read = root + source + '\\'
    write = root + target + '\\'
    a = np.load(method2)
    b = np.load(method1)
    c = np.load(method3)
    kon = np.zeros((10, 4), dtype='int32')

    for i in range(10):
        max = 0
        kon[i, 1] = i
        for j in range(10):
            for k in range(10):
                m = (b == i)
                n = (a == j)
                l = (c == k)
                h = (m * n)
                h = (h * l)
                if sum(h) > max:
                    max = sum(h)
                    kon[i, 0] = j
                    kon[i, 2] = k
                    kon[i, 3] = max
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    t = 0
    for file in listing:
        if (a[t] == kon[b[t], 0] and c[t] == kon[b[t], 2]):
            im = Image.open(read + file)
            if not os.path.exists(write+ str(b[t])):
                os.makedirs(write+ str(b[t]))
            try:
                im.save(write + str(b[t]) + "\\" + file)
            except IOError:
                print("cannot convert", read + file)
        t = t + 1

def simil(X, A, B):
    for i in range(0, 10):
        m = (X == i)
        print(i, "Novi i",sum(m))
        for j in range(10):
            n = (A[m] == j)
            t=(A==j)
            n = sum(n)
            print("Metoda2", n, j,n/sum(t))
        for k in range(10):
            l = (B[m] == k)
            l = sum(l)
            t = (B == k)
            print("Metoda3", l, k,l/sum(t))

def percent(X, A):
    _sum=0
    for i in range(0, max(X)+1):
        m = (X == i)
        _max=0
        for j in range(10):
            n = (A[m] == j)
            n = sum(n)
            if n>_max:
                _max=n
        _sum=_sum+_max
    return _sum/X.__len__()