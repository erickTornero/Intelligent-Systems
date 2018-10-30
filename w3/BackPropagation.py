import numpy as np
class Neuron:
    def __init__(self, NumberInputs, alpha):
        self.W = np.zeros(NumberInputs).reshape(NumberInputs, 1)
        self.bias = np.random.random(1)
        self.NumberInputs = NumberInputs
        self.alpha = alpha
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def GetOutput(self, X):
        r = np.matmul(self.W.T, X) + self.bias
        return self.sigmoid(r)

    def UpdateWeights(self):
        pass



class Layer:
    def __init__(self, NumberNeurons, NumberInputs, alpha):
        self.Neurons = [Neuron(NumberInputs, alpha)]*NumberNeurons
        self.NumberNeurons = NumberNeurons;

    def LayerOutput(self, X_in):
        Outs = []
        for neuron in self.Neurons:
            Outs.append(neuron.GetOutput(X_in))
        
        return np.array(Outs)


class NeuralNetwork:
    def __init__(self, d, NumIn, alpha = 0.001):
        self.Layers = [0]*len(d)
        self.alpha = alpha
        for key in d:
            if(key == 0):
                self.Layers[key] = Layer(NumberNeurons = d[key], NumberInputs = NumIn, alpha = alpha)
            elif(key > 0):
                self.Layers[key] = Layer(NumberNeurons = d[key], NumberInputs = d[key -1], alpha = alpha)

    def propagate(self, X_in, index = 0):
        X_t = np.array(0)
        for index in range(len(self.Layers)):
            if index == 0:
                X_t = self.Layers[index].LayerOutput(X_in)
            else:
                X_t = self.Layers[index].LayerOutput(X_t)
        return float(X_t)



dd = {0:10, 1:20, 2:1}
NN = NeuralNetwork(dd, 10, 0.05)