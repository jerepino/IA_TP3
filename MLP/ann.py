from time import sleep
import numpy as np
import csv
from matplotlib import pyplot as plt


class Capa:
    def __init__(self, n, input, act_f='relu'):
        """
        Clase para crear una capa
        :param n: cantidad de neuronas
        :param input: cantidad de entradas de la capa anterior (conexion por neurona)
        :param act_f: funcion de activacion de la capa
        """
        # np.random.seed(101)
        self.Wji = np.random.randn(n, input) / np.sqrt(input)
        self.bj = np.ones((n, 1)) * 0.1
        f = dict(relu=lambda x: np.maximum(x, 0), tanh=lambda x: np.tanh(x), sigmoid=lambda x: 1 / (1 + np.exp(-x)),
                 softplus=lambda x: np.log(1 + np.exp(x)), identidad=lambda x: x)
        df = dict(relu=lambda x: 1. * (x > 0), tanh=lambda x: 1 - np.tanh(x) ** 2,
                  sigmoid=lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2, softplus=lambda x: 1 / (1 + np.exp(-x)),
                  identidad=lambda x: np.ones_like(x))
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
        # print("\n\n Forward\n\n")
        X_ = X[:, np.newaxis]

        for l in range(0, len(self.capa)):
            if l == 0:
                a = self.capa[l].Wji @ X_ - self.capa[l].bj
            else:
                a = self.capa[l].Wji @ self.capa[l - 1].output - self.capa[l].bj
            self.capa[l].output = self.capa[l].g(a)
        return self.capa[-1].output

    def backpropagation(self, data_X, data_Y, lr=0.1):
        # print("\n\n Backpropagation\n\n")
        data_Y = data_Y[:, np.newaxis]
        X_ = data_X[:, np.newaxis]

        delta_ku = []
        for l in reversed(range(0, len(self.capa))):

            if l == 0:
                h_ku = self.capa[l].Wji @ X_ - self.capa[l].bj
            else:
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

    def train(self, train_X, train_Y, epoch, validation=False, vali_X=0, vali_Y=0, lr=0.1):
        err_train = []
        err_train_mean = []
        e_ = []
        if validation:
            err_vali = []
            err_vali_mean = []
        for e in range(epoch):

            for index, X in enumerate(train_X):
                z = self.forward(X)
                self.backpropagation(X, train_Y[index], lr)
                err_train.append(np.power(z - train_Y[index], 2))

            err_train_mean.append(np.mean(err_train))
            e_.append(e)

            if validation:
                for i, Val in enumerate(vali_X):
                    tk = self.forward(Val)
                    err_vali.append(np.power(tk - vali_Y[i], 2))
                err_vali_mean.append(np.mean(err_vali))

            if (e % 50 == 0) and (e != 0):
                plt.figure(1)
                plt.title('Error medio')
                plt.ylabel('Error')
                plt.xlabel('Epoch')
                plt.plot(e_, err_train_mean, 'b-.', label="Train")
                plt.legend()
                plt.grid(True)

                if validation:
                    plt.plot(e_, err_vali_mean, 'r', label="Validation")
                    plt.legend()

                # plt.ion()
                plt.show()
                plt.pause(.0001)
                if min(err_vali_mean) != err_vali_mean[-1]:
                    print("Empieza a haber overfiting, utilizo parada temprana")
                    break
            # print("El numero de iteraciones de train fueron: ", index)
            #     sleep(5)
            #     plt.close(1)

    def test(self, test_X, test_Y):
        err_test = []
        err_test_mean = []
        for i, Te in enumerate(test_X):
            tk = self.forward(Te)
            err_test.append(np.power(tk - test_Y[i], 2))
        err_test_mean.append(np.mean(err_test))
        print("El error cuadratico medio para nuestra red obtenido con el conjunto test es: ", err_test_mean)


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
    data = np.array(data).astype('float64')

    # ---------------------------------------------
    # -     Seleccion train, test y validation    -
    # ---------------------------------------------
    n_data = len(data)
    n_vali = int(n_data * 0.1)
    n_test = n_vali + int(n_data * 0.1)

    validation = data[:n_vali, :]
    test = data[n_vali:n_test, :]
    train = data[n_test:, :]

    # ---------------------------------------------
    # -     Normalizacion del dataset y test      -
    # ---------------------------------------------
    mean = np.mean(train, axis=0)
    # print("mean", mean)
    train -= mean
    test -= mean
    validation -= mean

    std = np.std(train, axis=0)
    # print("std", std)
    train /= std
    test /= std
    validation /= std

    # --------------------------------------
    # - Separo los datos de los resultados -
    # --------------------------------------
    train_X = np.array(train[:, :5])
    train_Y = np.array(train[:, -1])
    train_Y = train_Y[:, np.newaxis]

    validation_X = np.array(validation[:, :5])
    validation_Y = np.array(validation[:, -1])
    validation_Y = validation_Y[:, np.newaxis]

    test_X = np.array(test[:, :5])
    test_Y = np.array(test[:, -1])
    test_Y = test_Y[:, np.newaxis]

    epoch = 101
    net = Red()
    net.add(42, 5, act_f='relu')
    # net.add(25, 15, act_f='relu')
    net.add(1, 42, act_f='identidad')
    # net.add(1, 6, act_f='identidad')

    # print(net.capa[0].Wji)

    net.train(train_X, train_Y, epoch, True, validation_X, validation_Y, 0.0005)
    # net.test(test_X, test_Y)
