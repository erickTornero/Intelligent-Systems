# Generate data with Gaussian distribution:

def getBlobsData(samples = 200, ncenters = 3, rstate = 40):
    from sklearn.datasets.samples_generator import make_blobs
    X, Y = make_blobs(n_samples=samples, centers=ncenters, n_features=2, random_state=rstate)
    return X, Y

# Plot Scatter data
import matplotlib.pyplot as plt
def plot(X, y, colors):
    
    import pandas as pd
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    _, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()

# Plot mesh & contour of SVM classification:



# Getting the data
X, Y = getBlobsData()
#plot(X,Y, colors = {0:'red', 1:'green',2:'blue'})
print(X.shape)
print(Y.shape)
# Get training & test data:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# Return a trained model The model:
def TrainSVMModel(X, Y,  C_ = 1.0, kindofkernel = 'linear', gamma = 0):
    from sklearn.svm import SVC
    if kindofkernel == 'rbf':
        model = SVC( kernel = kindofkernel, C = C_, gamma= gamma)
    else:
    	model = SVC( kernel = kindofkernel, C = C_)
    model.fit(X, Y)
    return model

# Create Diferents models
linearModel = TrainSVMModel(X_train, Y_train, C_=0.1,kindofkernel='linear')
polinomModel = TrainSVMModel(X_train, Y_train, C_ = 0.1, kindofkernel='poly')
gaussianModel = TrainSVMModel(X_train, Y_train, C_= 12, kindofkernel='rbf', gamma = 10)

models = (linearModel, polinomModel, gaussianModel)

from PlotMeshSVM import get_meshgrid, plot_contours
titles = ('linear', 'polynomical', 'Gaussian')
import matplotlib.pyplot as plt
fig, sub = plt.subplots(3, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = get_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# Evaluate performances of models:
Y_pred = linearModel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
