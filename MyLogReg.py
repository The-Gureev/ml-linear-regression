import numpy as np
import pandas as pd
import math

class MyLogReg:
    def __init__(self, n_iter = 10, learning_rate = 0.1, weights = None, metric = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        
    def __str__(self):
	    return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calc_metric(self, y_true, y_predict):
        if (self.metric == 'accuracy'):
            correct_predict = np.sum(y_true = y_predict)
            total = len(y_predict)
            return correct_predict / total
        
    def create_matrix(self, data):
        data_with_bias = data.copy()
        data_with_bias.insert(0, 'bias', 1)
        return data_with_bias.values

    def fit(self, X, y, verbose):
        X_matrix = self.create_matrix(X)
        y_vector = y.values

        n_samples = X_matrix.shape[0]
        n_features = X_matrix.shape[1]

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

    def predict_proba(self, X):
        X_matrix = self.create_matrix(X)

        return self.sigmoid(X_matrix @ self.weights)

    def predict(self, X):
        return self.predict_proba(X) > 0.5






X = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9,10]})
y = pd.Series([0,1,2,3,4,5,6,7,8,9,10])

log_reg = MyLogReg()
log_reg.fit(X, y, True)



