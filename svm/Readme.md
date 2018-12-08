Descargar los archivos *PlotMeshSVM.py* y *svm_model.py*

En la linea de comando con python3 ejecutar el siguiente comando:

**python svm_model.py**

Existen 3 funciones donde se configuran los kernels de los modelos

*linearModel = TrainSVMModel(X_train, Y_train, C_=0.01,kindofkernel='linear')*
*polinomModel = TrainSVMModel(X_train, Y_train, C_ = 0.01, kindofkernel='poly')*
*gaussianModel = TrainSVMModel(X_train, Y_train, C_= 0.1, kindofkernel='rbf')*
