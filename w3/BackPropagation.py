import numpy as np
np.random.seed(20)
class Neuron:
    def __init__(self, NumberInputs, alpha, isInput):
        # self.W = np.zeros(NumberInputs).reshape(NumberInputs, 1)
        if isInput == False:
            self.W = np.random.rand(NumberInputs).reshape(NumberInputs, 1)
            self.bias = np.random.random(1)
            self.NumberInputs = NumberInputs
            self.alpha = alpha
    

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def GetOutput(self, X):
        r = np.matmul(self.W.T, X) + self.bias
        return float(self.sigmoid(r))

    def UpdateWeights(self):
        pass



class Layer:
    def __init__(self, NumberNeurons, NumberInputs, alpha, isInput = False):
        self.Neurons = [Neuron(NumberInputs, alpha, isInput) for i in range(NumberNeurons)]
        self.NumberNeurons = NumberNeurons
        self.isInput = isInput

    def LayerOutput(self, X_in):
        Outs = []
        if self.isInput == False:
            for neuron in self.Neurons:
                Outs.append(neuron.GetOutput(X_in))
            return np.array(Outs).reshape(len(Outs), 1)
        else:
            # Ws = np.array([neuron.W for neuron in self.Neurons])
            # print(Ws)
            return X_in


class NeuralNetwork:
    def __init__(self, d, NumIn, alpha = 0.001):
        self.Layers = [0]*len(d)
        self.alpha = alpha
        for key in d:
            if(key == 0):
                self.Layers[key] = Layer(NumberNeurons = d[key], NumberInputs = NumIn, alpha = alpha, isInput=True)
            elif(key > 0):
                self.Layers[key] = Layer(NumberNeurons = d[key], NumberInputs = d[key -1], alpha = alpha, isInput=False)

    def propagate(self, X_in):
        X_t = np.array(0)
        L = []
        for index in range(len(self.Layers)):
            if index == 0:
                X_t = self.Layers[index].LayerOutput(X_in)
            else:
                X_t = self.Layers[index].LayerOutput(X_t)
            
            L.append(X_t)
        return (X_t, L)
    
    def backward(self, Outs, y):
        y_hat = Outs[-1]
        for i in range(len(Outs) - 1):
            index = -1 - i
            X_in = Outs[index -1]
            
            #delta = (y-y_hat)*y_hat*(np.ones(y_hat.shape[0]).reshape(y_hat.shape[0],1) - y_hat)
            #deltaW = X_in*delta.T
            #deltaW.shape
            #for j in range(deltaW.shape[1]):
            #    self.Layers[index].Neurons[j].W = self.Layers[index].Neurons[j].W - self.alpha*deltaW[j]
            
            #print(deltaW, deltaW.shape)
            delta = (y_hat-y)
            deltaB = y_hat - y
            for j in range(i):
                delta = delta*Outs[-2-j]*(1 - Outs[-2-j])
                W = [self.Layers[-1-j].Neurons]
                deltaB = delta*Outs[-2-j]*(1 - Outs[-2-j])*W0
            delta = X_in*delta.T
            #print(delta.shape)
            for j in range(self.Layers[index].NumberNeurons):
                self.Layers[index].Neurons[j].W = self.Layers[index].Neurons[j].W - delta[:,j].reshape(delta.shape[0],1)*self.alpha

    def train(self, X, Y, ephocs = 10):
        rows, cols = X.shape
        for ep in range(ephocs):
            error = 0
            for i in range(rows):
                x = X[i,:].reshape(cols, 1)
                y = Y[i].reshape(1,1)
                y_hat, Outs = self.propagate(x)
                self.backward(Outs, y)
                error = error + (y_hat - y)*(y_hat - y)
            print(error)

dd = {0:4, 1:2, 2:1}
NN = NeuralNetwork(dd, 10, 0.05)
inp = np.ones(4).reshape(4,1)
ans = NN.propagate(inp)
print('Answer:>',ans[0], type(ans[0]), ans[0].shape )
#print(ans[1])
Y = np.array([0])
NN.backward(ans[1],Y.reshape(1,1))
xx = 1
