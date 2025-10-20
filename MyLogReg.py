import numpy as np 
import pandas as pd

class MyLogReg:
    def __init__(self, n_iter = 10, learning_rate = 0.1, weights = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        
    def __str__(self):
	    return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X, y, verbose):
        X.insert(0, 'bias', 1)
        X_matrix = X.copy().values
        y_vector = y.values

        l_features = len(X_matrix)


        for i in range(self.n_iter):
            print(i)

X = pd.DataFrame({'X':[0,1,2,3,4,5,6,7,8,9]})
y = pd.Series([0.1,1.2,2.1,3.2,4.3,5.4,6.5,7.6,8.7,9.1])

myLogReg = MyLogReg()
myLogReg.fit(X, y, 10)

