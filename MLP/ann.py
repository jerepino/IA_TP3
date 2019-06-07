import numpy as np
import csv
import matplotlib as pl


class Capa:
    def __init__(self, n, input, act_f='relu'):
        """
        Clase para crear una capa
        :param n: cantidad de neuronas
        :param input: cantidad de entradas de la capa anterior (conexion por neurona)
        :param act_f: funcion de activacion de la capa
        """
        np.random.seed(101)
        self.Wji = np.random.randn(n, input) / np.sqrt(n)
        self.bj = np.ones((n, 1)) * 0.1
        f = dict(relu=lambda x: np.maximum(x, 0), tanh=lambda x: np.tanh(x), sigmoid=lambda x: 1 / (1 + np.exp(-x)),
                 softplus=lambda x: np.log(1 + np.exp(x)))
        df = dict(relu=lambda x: (0, 1)[x >= 0], tanh=lambda x: 1 - np.tanh(x) ** 2,
                  sigmoid=lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2, softplus=lambda x: 1 / (1 + np.exp(-x)))
        self.g = f[act_f]
        self.dg = df[act_f]
        self.n = n
        self.output = []


class Red:
    def __init__(self):
        self.capa = []

    def add(self, n, p, act_f='relu'):
        self.capa.append(Capa(n, p, act_f))

    def forward(self, X):
        print("\n\n Forward\n\n")
        self.capa[0].output = X[:, np.newaxis]

        for l in range(1, len(self.capa)):
            a = self.capa[l].Wji @ self.capa[l - 1].output - self.capa[l].bj
            self.capa[l].output = self.capa[l].g(a)

            # print('En la capa :', l)
            # print("valor entrada ")
            # print(self.capa[l - 1].output)
            # print(" Wji")
            # print(self.capa[l].Wji)
            # print("bias")
            # print(self.capa[l].bj)
            # print(" output")
            # print(self.capa[l].output)

        return self.capa[-1].output

    def backpropagation(self, data_Y, lr=0.1):
        print("\n\n Backpropagation\n\n")
        data_Y = data_Y[:, np.newaxis]
        delta_ku = []
        for l in reversed(range(1, len(self.capa))):

            h_ku = self.capa[l].Wji @ self.capa[l - 1].output - self.capa[l].bj

            if l == (len(self.capa) - 1):

                delta_ku.insert(0, (data_Y - self.capa[l].g(h_ku)) @ self.capa[l].dg(h_ku))

            else:

                delta_ku.insert(0, delta_ku[0] @ W @ np.diagflat(self.capa[l].dg(h_ku)))

            W = self.capa[l].Wji.copy()

            dW = lr * delta_ku[0].T @ self.capa[l - 1].output.T
            dB = (-1) * lr * delta_ku[0].T

            self.capa[l].bj += dB
            self.capa[l].Wji += dW

            # print("En la capa", l)
            # print("Valor deseado - obtenido")
            # print(data_Y, self.capa[-1].output)
            # print("Wij")
            # print(self.capa[l].Wji)
            # print("bj")
            # print(self.capa[l].bj)

    def train(self, data_X, data_Y, epoc, data_val=0):
        error = []
        for e in range(epoc):
            for index, X in enumerate(data_X):
                z = self.forward(X)
                self.backpropagation(data_Y[index])
        # if index % 25 == 0:
        #     error.append(z-data_Y)
        #     pl.plot(error,e)

if __name__ == '__main__':

    # ---------------------------------
    # - Obtencio de datos del dataset -
    # ---------------------------------
    filename = 'USA_Housing.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data)
    x = list(reader)
    data = np.array(x)

    # ---------------------------------------------
    # - Eliminacion de la cabecera y la direccion -
    # ---------------------------------------------
    data = np.delete(data, 6, axis=1)
    data = np.delete(data, 0, axis=0)
    data = np.array(data).astype('float32')

    # ---------------------------------------------
    # -     Seleccion train, test y validation    -
    # ---------------------------------------------
    train = data[4000, :]
    test = data[4000:4500, :]
    validation = data[4500:, :]

    # ---------------------------------------------
    # -     Normalizacion del dataset y test      -
    # ---------------------------------------------
    for i, val in enumerate(train[0, :]):
        mean = np.mean(train[:, i])
        train[:, i] -= mean
        test[:, i] -= mean
        validation[:, i] -= mean
        std = np.std(train[:, i])
        train[:, i] /= std
        test[:, i] -= std
        validation[:, i] -= std

    # --------------------------------------
    # - Separo los datos de los resultados -
    # --------------------------------------
    train_X = np.array(train[:, :5])
    print(train_X.shape)
    train_Y = np.array(train[:, -1])
    train_Y = train_Y[:, np.newaxis]
    print(train_Y.shape)
    print(train_X)

# print(data_X.shape, data_Y.shape)
#  net = Red()
#  net.add(3, 3)
#  net.add(2, 3, act_f='tanh')
#  net.add(1, 2, act_f='tanh')
#  print("La topologia es 3 - 2 - 1")
#  net.train(data_X,data_Y,1)
