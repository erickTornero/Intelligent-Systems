import numpy as np
from numpy.linalg import inv
class MyLogisticRegression():
    def __init__(self, coef = None):
        self.coef = coef
    
    def logistic_prob(self, X, B):
        rows = np.shape(X)[0] # Numero de filas
        cols = np.shape(X)[1] # Número de columnas
        pi = list(range(1, rows + 1))
        exponent = list(range(1, rows +1 ))
        # Obtener las probabilidades:
        for i in range(rows):
            exponent[i] = 0
            # Obtener los exponentes, esto es por columnas:
            for j in range(cols):
                ex = X[i][j]*B[j]
                exponent[i] = exponent[i] + ex
            # End for exps
            with np.errstate(divide='ignore', invalid='ignore'):
                pi[i] = 1/(1 + np.exp(-exponent[i]))
        return pi

    def getW(self, P):
        n = len(P)
        W = np.zeros(n*n).reshape(n,n)
        for i in range(n):
            W[i,i] = P[i]*(1-P[i])
            W[i,i].astype(float)
        return W

    def fit(self, data, labels, err_allowed):
        X = data.values
        Y = labels.values
        rows = np.shape(X)[0]
        # Definición de la entrada bias, siempre es 1
        bias = np.ones(rows).reshape(rows, 1)
        # Add to the end of the array, Bias.
        __X = np.append(X, bias, axis = 1)
        cols = np.shape(__X)[1]
        # Inicializando beta como una matriz columna de ceros
        B = np.zeros(cols).reshape(cols, 1)
        # Primero se obtienen las probabilidades:
        ## range(1, t) itera desde 1 hasta t-1
        dB = np.array(range(1, cols + 1)).reshape(cols, 1)
        # Definir un error inicial
        current_error = 1000
        while current_error > err_allowed:
            # Obtener la matriz Pi
            Pi = []
            # Se obtiene una lista con todas las probabilidades
            Pi = self.logistic_prob(__X, B)
            # Obtener la matriz W:
            W = self.getW(Pi)
            den = inv(np.matmul(np.matmul(np.transpose(__X),W), __X))
            inter = (Y- np.transpose(Pi)).transpose()
            num = np.matmul(np.transpose(__X),(inter))
            dB = np.matmul(den, num)
            # Get the new Beta value
            B = B + dB
            current_error = np.sum(dB*dB)
            print('Current Error>', current_error)
            self.coef = B
        print('B>', B)
    
    def dotproduct(self, a, b):
        return sum(list(map(lambda x, y: x*y, a,b)))

    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))

    def predict(self, X_test, threshold):
        if(np.shape(self.coef)[0] == 0 and np.shape(self.coef)[1] == 0):
            print('Error: Entrenar el modelo')
        else:
            X = X_test.values
            W = self.coef[:-1]
            b = self.coef[-1]
            estimated = np.zeros(np.shape(X)[0]).reshape(np.shape(X)[0], 1)
            for i in range(0, np.shape(X)[0]):
                xi = X[i,:]
                reg = self.dotproduct(xi,W) + b
                prob = self.sigmoid(float(reg))
                if prob >= threshold:
                    estimated[i,0] = 1
                else:
                    estimated[i,0] = 0
            return estimated
                
    