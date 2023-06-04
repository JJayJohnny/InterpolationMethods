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
    # step = int(len(x)/interpolationNodesCount)
    # # wyznaczenie rownooddalonych wezlow interpolacji
    # for i in range(0, len(x), step):
    #     interpolationNodes.append(x[i])
    #     interpolationNodesValues.append(y[i])
    idx = np.round(np.linspace(0, len(x) - 1, interpolationNodesCount)).astype(int)
    interpolationNodes = x[idx]
    interpolationNodesValues = y[idx]

    yInter = []
    xInter = []

    # for xI in np.linspace(0, x[len(x)-1], 1000):
    # interpolujemy w tych samych punktach co znane wartosci aby moc porownac jakosc interpolacji
    for xI in x:
        fx = 0
        for i in range(0, len(interpolationNodes)):
            fx += interpolationNodesValues[i] * fi(i, xI, interpolationNodes)
        yInter.append(fx)
        xInter.append(xI)

    return np.array(xInter), np.array(yInter)


def splineMethod(x, y, interpolationNodesCount):
    interpolationNodes = []
    interpolationNodesValues = []
    # step = int(x.shape[0] / interpolationNodesCount)
    # # wyznaczenie rownooddalonych wezlow interpolacji
    # for i in range(0, len(x), step):
    #     interpolationNodes.append(x[i])
    #     interpolationNodesValues.append(y[i])
    idx = np.round(np.linspace(0, len(x) - 1, interpolationNodesCount)).astype(int)
    interpolationNodes = x[idx]
    interpolationNodesValues = y[idx]

    A = np.zeros((4*(interpolationNodesCount-1), 4*(interpolationNodesCount-1)))
    Y = np.zeros((4*(interpolationNodesCount-1), 1))

    row = 0
    # wpisanie wartosci na poczatkach przedzialow
    for i in range(0, interpolationNodesCount-1):
        A[row, 4*i] = 1
        Y[row] = interpolationNodesValues[i]
        row += 1

    # wpisanie wartosci na koncach przedzialow
    for i in range(1, interpolationNodesCount):
        h = interpolationNodes[i] - interpolationNodes[i-1]
        A[row, (i-1)*4 + 0] = 1
        A[row, (i-1)*4 + 1] = h
        A[row, (i-1)*4 + 2] = pow(h, 2)
        A[row, (i-1)*4 + 3] = pow(h, 3)
        Y[row] = interpolationNodesValues[i]
        row += 1

    # pierwsze pochodne
    for i in range(1, interpolationNodesCount-1):
        h = interpolationNodes[i] - interpolationNodes[i - 1]
        A[row, (i-1)*4 + 1] = 1
        A[row, (i-1)*4 + 2] = 2 * h
        A[row, (i-1)*4 + 3] = 3 * pow(h, 2)
        A[row, i*4 + 1] = -1
        Y[row] = 0
        row += 1

    # drugie pochodne
    for i in range(1, interpolationNodesCount-1):
        h = interpolationNodes[i] - interpolationNodes[i - 1]
        A[row, (i-1)*4 + 2] = 2
        A[row, (i-1)*4 + 3] = 6 * h
        A[row, i*4 + 2] = -2
        Y[row] = 0
        row += 1

    # zerowanie drugich pochodnych na poczatku i koncu
    A[row, 2] = 1
    Y[row] = 0
    row += 1
    h = interpolationNodes[len(interpolationNodes) - 1] - interpolationNodes[len(interpolationNodes) - 2]
    A[row, A.shape[1]-2] = 2
    A[row, A.shape[1]-1] = 6 * h
    Y[row] = 0

    parameters = np.linalg.solve(A, Y)
    print(np.allclose(np.dot(A, parameters), Y))
    split = -1
    yInter = []
    xInter = []
    for xI in x:
        if xI in interpolationNodes:
            split += 1
            if split >= interpolationNodesCount - 1:
                split -= 1
        h = xI - interpolationNodes[split]
        fx = parameters[split*4, 0] + parameters[split*4+1, 0]*h + parameters[split*4+2, 0]*pow(h, 2) + parameters[split*4+3, 0]*pow(h, 3)
        xInter.append(xI)
        yInter.append(fx)


    return np.array(xInter), np.array(yInter)




if __name__ == '__main__':
    data = pd.read_csv(dataPath, header=0, names=['x', 'y'])

    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    # x = np.array([1, 3, 5])
    # y = np.array([6, -2, 4])

    xInter, yInter = lagrangeMethod(x, y, 40)

    plt.plot(x, y)
    plt.plot(xInter[100:-100], yInter[100:-100])
    plt.show()
