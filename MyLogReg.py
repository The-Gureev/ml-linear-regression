import numpy as np
import pandas as pd
import math

class MyLogReg:
    def __init__(self, n_iter = 10, learning_rate = 0.1, weights = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        
    def __str__(self):
	    return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, verbose):
        X_with_bias = X.copy() #.insert(1, 'bias', 1)
        X_with_bias.insert(0, 'bias', 1)
        y_vector = y.values

        X_matrix = X_with_bias.values
        n_samples = X_matrix.shape[0]
        n_features = X_with_bias.shape[1]

        self.weights = np.ones(n_features)
       
        eps = 1e-15

        predict = self.sigmoid(X_matrix @ self.weights)
        log_loss = -1*np.mean( y_vector * np.log(predict + eps) + (1-y_vector)*np.log(1-predict + eps) )

        if verbose:
            print(f'start | loss: {log_loss:.2f}')

        for rate in range(self.n_iter):
            predict = self.sigmoid(X_matrix @ self.weights)
            
            log_loss = -1*np.mean( y_vector * np.log(predict + eps) + (1-y_vector)*np.log(1-predict + eps) )
            gradient = (X_matrix.T @ (predict - y_vector)) / n_samples
            self.weights -= self.learning_rate * gradient
            if verbose and rate % verbose == 0:
                print(f'{rate} | loss: {log_loss:.2f}')
    
    def get_coef(self):
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.weights[1:]


X = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9,10]})
y = pd.Series([0,1,2,3,4,5,6,7,8,9,10])

log_reg = MyLogReg()
log_reg.fit(X, y, True)



