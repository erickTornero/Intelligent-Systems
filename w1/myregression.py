
import pandas as pd
from numpy.linalg import inv
from numpy import matmul, transpose

def load_data(url_d = None, index_c = 0):
    if url_d == None:
        print("Error: Ingrese Nombre del archivo o dirección")
    else:
        return pd.read_csv(url_d, index_col = index_c)


class MyLinearRegression:
    def __init__(self, coef = None, bias = None):
        self.coef = coef
        self.bias = bias
    
    def fit(self, data, labels):
        X = data.values
        Y = labels.values
        B = inv(matmul(transpose(X),X))
        B = matmul(B, matmul(transpose(X),Y))
        self.coef = B
        self.bias = B[0]
    def predict(self, X):
        X_ = X.values
        return matmul(X_, self.coef)