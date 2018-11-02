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

    def propagate(self, X_in, justAnswer = False):
        X_t = np.array(0)
        L = []
        for index in range(len(self.Layers)):
            if index == 0:
                X_t = self.Layers[index].LayerOutput(X_in.T)
            else:
                X_t = self.Layers[index].LayerOutput(X_t)
            
            L.append(X_t)
        if justAnswer:
            return X_t
        else:
            return (X_t, L)
    
    def backward(self, Outs, y):
        y_hat = Outs[-1]
        deltaB = np.array(y_hat - y).reshape(1,1)
        delta = np.array(y_hat - y).reshape(1,1)*Outs[-2]
        for i in range(len(Outs) - 1):
            index = -1 - i
            X_in = Outs[index -1].copy()
            
            #curr_Wlen = self.Layers[index].Neurons[0].W.shape[0]
            #currW = np.ones(curr_Wlen).reshape(curr_Wlen, 1)
            #for k in range(self.Layers[index+1].NumberNeurons):
            #    currW = np.append(currW, self.Layers[index+1].Neurons[k].W, axis = 1)
            #currW = np.delete(currW, 0, axis = 1)

            #delta = (y-y_hat)*y_hat*(np.ones(y_hat.shape[0]).reshape(y_hat.shape[0],1) - y_hat)
            #deltaW = X_in*delta.T
            #deltaW.shape
            #for j in range(deltaW.shape[1]):
            #    self.Layers[index].Neurons[j].W = self.Layers[index].Neurons[j].W - self.alpha*deltaW[j]
            
            #print(deltaW, deltaW.shape)
            #delta = (y_hat-y)
            if index != -1:
                wlength = self.Layers[index+1].Neurons[0].W.shape[0]
                Ws = np.ones(wlength).reshape(wlength, 1)
                for k in range(self.Layers[index+1].NumberNeurons):
                    Ws = np.append(Ws, self.Layers[index+1].Neurons[k].W, axis = 1)
                Ws = np.delete(Ws, 0, axis = 1)
                #print(Ws, Ws.shape)
                deltaB = np.sum(deltaB.T * Ws, axis = 1).reshape(Ws.shape[0], 1)
                deltaB = deltaB * Outs[index]
                #print(deltaB, deltaB.shape)
                #print(delta.shape, Ws.shape, Outs[index].shape)
                delta = np.sum(delta*Ws*(1 -Outs[index]), axis = 1).reshape(Ws.shape[0],1)
                delta = delta.T*X_in
                #print('DeltaW>\n',delta, delta.shape)
            for j in range(i):
                pass
                #delta = delta*Outs[-2-j]*(1 - Outs[-2-j])
                
                #print('W info>\n',Ws, Ws.shape)
                #deltaB = deltaB*Outs[index]
                #deltaB = delta*Outs[-2-j]*(1 - Outs[-2-j])*W0
            #delta = X_in*delta.T
            #print(delta.shape)
            for j in range(self.Layers[index].NumberNeurons):
                self.Layers[index].Neurons[j].W = self.Layers[index].Neurons[j].W - delta[:,j].reshape(delta.shape[0],1)*self.alpha
                self.Layers[index].Neurons[j].bias = self.Layers[index].Neurons[j].bias - self.alpha*deltaB[j]

    def train(self, X, Y, ephocs = 10):
        rows, cols = X.shape
        for _ in range(ephocs):
            error = 0
            for i in range(rows):
                x = X[i,:].reshape(1, cols)
                y = Y[i].reshape(1,1)
                y_hat, Outs = self.propagate(x)
                self.backward(Outs, y)
                error = error + (y_hat - y)*(y_hat - y)
            print(error)
    
    def predict_one(self, X):
        a = self.propagate(X, justAnswer=True)
        if(a > 0.5):
            a = 1
        else:
            a = 0
        return a
    def predictSet(self, X):
        L = []
        for i in range(X.shape[0]):
            L.append(self.predict_one(X[i,:]))
        return np.array(L).reshape(len(L),1)

dd = {0:4, 1:4, 2:2,3:1}
NN = NeuralNetwork(dd, 4, 0.05)
inp = np.ones(4).reshape(1,4)
ans = NN.propagate(inp)
print('Answer:>',ans[0], type(ans[0]), ans[0].shape )
print(ans[1])
Y = np.array([0])

NN.backward(ans[1],Y.reshape(1,1))
xx = 1
