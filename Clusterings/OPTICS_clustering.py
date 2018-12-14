import numpy as N
import pylab as P
from scipy.spatial.distance import squareform,pdist

def optics(x, k, distMethod='euclidean'):
    if len(x.shape) > 1:
        m, n = x.shape
    else:
        m = x.shape[0]
        n = 1

    try:
        D = squareform(pdist(x, distMethod))
        distOK = True
    except:
        print
        "squareform or pdist error"
        distOK = False

    CD = N.zeros(m)
    RD = N.ones(m) * 1E10

    for i in range(m):
        # again you can use the euclid function if you don't want hcluster
        #        d = euclid(x[i],x)
        #        d.sort()
        #        CD[i] = d[k]

        tempInd = D[i].argsort()
        tempD = D[i][tempInd]
        #        tempD.sort() #we don't use this function as it changes the reference
        CD[i] = tempD[k]  # **2

    order = []
    seeds = N.arange(m, dtype=N.int)

    ind = 0
    while len(seeds) != 1:
        #    for seed in seeds:
        ob = seeds[ind]
        seedInd = N.where(seeds != ob)
        seeds = seeds[seedInd]

        order.append(ob)
        tempX = N.ones(len(seeds)) * CD[ob]
        tempD = D[ob][seeds]  # [seeds]
        # you can use this function if you don't want to use hcluster
        # tempD = euclid(x[ob],x[seeds])

        temp = N.column_stack((tempX, tempD))
        mm = N.max(temp, axis=1)
        ii = N.where(RD[seeds] > mm)[0]
        RD[seeds[ii]] = mm[ii]
        ind = N.argmin(RD[seeds])

    order.append(seeds[0])
    RD[0] = 0  # we set this point to 0 as it does not get overwritten
    return RD, CD, order


def euclid(i, x):
    """euclidean(i, x) -> euclidean distance between x and y"""
    y = N.zeros_like(x)
    y += 1
    y *= i
    if len(x) != len(y):
        raise(ValueError, "vectors must be same length")

    d = (x - y) ** 2
    return N.sqrt(N.sum(d, axis=1))


if __name__ == "__main__":
    testX = N.array([[15., 70.],
                     [31., 87.],
                     [45., 32.],
                     [5., 8.],
                     [73., 9.],
                     [32., 83.],
                     [26., 50.],
                     [7., 31.],
                     [43., 97.],
                     [97., 9.]])

    #    mlabOrder = N.array(1,2,6,7,3,8,9,4,5,10) #the order returned by the original MATLAB code
    # Remeber MATLAB counts from 1, python from 0


    P.plot(testX[:, 0], testX[:, 1], 'ro')
    RD, CD, order = optics(testX, 4)
    testXOrdered = testX[order]
    P.plot(testXOrdered[:, 0], testXOrdered[:, 1], 'b-')

    print(order)

    P.show()