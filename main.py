import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataPath = 'data/'
plotPath = 'plots/'

# TODO:
# no mówił o sprawku, że minimum 3 trasy i do każdej z tych tras minimum 4 wykresy po dwa na metodę


def fi(i, x, interpolationNodes):
    nominator = 1
    denominator = 1
    for j in range(len(interpolationNodes)):
        if i != j:
            nominator *= (x - interpolationNodes[j])
            denominator *= (interpolationNodes[i] - interpolationNodes[j])
    return nominator/denominator


def lagrangeMethod(x, y, interpolationNodesCount, random=False):
    idx = np.round(np.linspace(0, len(x) - 1, interpolationNodesCount)).astype(int)
    if random:
        idx = np.random.choice(x.shape[0], interpolationNodesCount, replace=False)
        idx = np.sort(idx)
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

    return np.array(xInter), np.array(yInter), np.array(interpolationNodes), np.array(interpolationNodesValues)


def splineMethod(x, y, interpolationNodesCount, random=False):
    idx = np.round(np.linspace(0, len(x) - 1, interpolationNodesCount)).astype(int)
    if random:
        idx = np.random.choice(x.shape[0], interpolationNodesCount, replace=False)
        idx = np.sort(idx)
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

    return np.array(xInter), np.array(yInter), np.array(interpolationNodes), np.array(interpolationNodesValues)


def div(x, y, interpolationFunction, kMin, kMax):
    k = []
    divK = []

    xI, yI = interpolationFunction(x, y, kMin-1)
    prev = yI

    for i in range(kMin, kMax):
        xI, yI, interpolationNodes, interpolationNodesValues = interpolationFunction(x, y, i)
        d = np.max(np.max(np.abs(yI - prev)))
        k.append(i)
        divK.append(d)
        prev = yI

    return np.array(k), np.array(divK)


def meanSquareError(x, y, interpolationFunction, kMin, kMax):
    k = []
    MSE = []

    for i in range(kMin, kMax):
        xI, yI, interpolationNodes, interpolationNodesValues = interpolationFunction(x, y, i)
        err = np.sqrt(np.sum(np.power(y - yI, 2)))/y.shape[0]
        k.append(i)
        MSE.append(err)

    return np.array(k), np.array(MSE)


def makePlots(x, y, dataName, k=15):
    # wykres samej trasy
    plt.figure()
    plt.plot(x, y, label="Dane rzeczywiste")
    plt.title(f"Trasa: {dataName}", loc='center', wrap=True)
    plt.xlabel("Dystans [m]")
    plt.ylabel("Wysokość [m]")
    plt.savefig(f"{plotPath}{dataName}.png")

    xL, yL, interpolationNodes, interpolationNodesValues = lagrangeMethod(x, y, k)
    # metoda Lagrange'a
    plt.figure()
    plt.plot(x, y, label="Dane rzeczywiste")
    plt.plot(xL, yL, label="Metoda Lagrange'a")
    plt.scatter(interpolationNodes, interpolationNodesValues, label="Węzły interpolacji", color="red")
    plt.title(f"{dataName}: interpolacja Lagrange'a dla K={k}", loc='center', wrap=True)
    plt.xlabel("Dystans [m]")
    plt.ylabel("Wysokość [m]")
    plt.legend()
    plt.savefig(f"{plotPath}{dataName}_Lagrange.png")

    # metoda Lagrange'a przyblizona
    plt.figure()
    plt.plot(x, y, label="Dane rzeczywiste")
    plt.plot(xL[50:-50], yL[50:-50], label="Metoda Lagrange'a")
    plt.scatter(interpolationNodes, interpolationNodesValues, label="Węzły interpolacji", color="red")
    plt.title(f"{dataName}: metoda Lagrange'a dla K={k} z pominięciem 50 wartości na początku i końcu", loc='center', wrap=True)
    plt.xlabel("Dystans [m]")
    plt.ylabel("Wysokość [m]")
    plt.legend()
    plt.savefig(f"{plotPath}{dataName}_Lagrange_Zoom.png")

    xS, yS, interpolationNodes, interpolationNodesValues = splineMethod(x, y, k)
    # funkcje sklejane
    plt.figure()
    plt.plot(x, y, label="Dane rzeczywiste")
    plt.plot(xS, yS, label="Metoda fukncji sklejanych")
    plt.scatter(interpolationNodes, interpolationNodesValues, label="Węzły interpolacji", color="red")
    plt.title(f"{dataName}: metoda funkcji sklejanych trzeciego stopnia dla K={k}", loc='center', wrap=True)
    plt.xlabel("Dystans [m]")
    plt.ylabel("Wysokość [m]")
    plt.legend()
    plt.savefig(f"{plotPath}{dataName}_Spline.png")

    # blad sredniokwadratowy dla spline
    k, MSE = meanSquareError(x, y, splineMethod, 2, 50)

    plt.figure()
    plt.semilogy(k, MSE)
    plt.title(f"{dataName}: błąd średniokwadratowy dla metody funkcji sklejanych", loc='center', wrap=True)
    plt.xlabel("Liczba węzłów interpolacji")
    plt.ylabel("MSE")
    plt.savefig(f"{plotPath}{dataName}_Spline_MSE.png")

    # blad sredniokwadratowy dla lagrange
    k, MSE = meanSquareError(x, y, lagrangeMethod, 2, 50)

    plt.figure()
    plt.semilogy(k, MSE)
    plt.title(f"{dataName}: błąd średniokwadratowy dla metody Lagrange'a", loc='center', wrap=True)
    plt.xlabel("Liczba węzłów interpolacji")
    plt.ylabel("MSE")
    plt.savefig(f"{plotPath}{dataName}_Lagrange_MSE.png")



if __name__ == '__main__':
    data = pd.read_csv(dataPath+"SpacerniakGdansk.csv", header=0, names=['x', 'y'])

    x = data['x'].to_numpy()
    y = data['y'].to_numpy()

    makePlots(x, y, "SpacerniakGdansk", 20)
