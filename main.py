import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataPath = 'data/SpacerniakGdansk.csv'

#TODO:
# no mówił o sprawku, że minimum 3 trasy i do każdej z tych tras minimum 4 wykresy po dwa na metodę


def fi(i, x, interpolationNodes):
    nominator = 1
    denominator = 1
    for j in range(len(interpolationNodes)):
        if i != j:
            nominator *= (x - interpolationNodes[j])
            denominator *= (interpolationNodes[i] - interpolationNodes[j])
    return nominator/denominator


def lagrangeMethod(x, y, interpolationNodesCount):
    interpolationNodes = []
    interpolationNodesValues = []
    step = int(len(x)/interpolationNodesCount)
    for i in range(0, len(x), step):
        interpolationNodes.append(x[i])
        interpolationNodesValues.append(y[i])

    yInter = []
    xInter = []

    # for xI in np.linspace(0, x[len(x)-1], 1000):
    for xI in x:
        fx = 0
        for i in range(0, len(interpolationNodes)):
            fx += interpolationNodesValues[i] * fi(i, xI, interpolationNodes)
        yInter.append(fx)
        xInter.append(xI)

    return xInter, yInter





if __name__ == '__main__':
    data = pd.read_csv(dataPath, header=0, names=['x', 'y'])

    x = data['x'].to_numpy()
    y = data['y'].to_numpy()

    xInter, yInter = lagrangeMethod(x, y, 15)

    plt.plot(x, y)
    plt.plot(xInter, yInter)
    plt.show()
