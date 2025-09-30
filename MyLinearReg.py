import numpy as np
import pandas as pd

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate = 0.1, weights = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
    
    def __str__(self):
	    return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def _initialize_weights(self, x):
        num_features = len(x)
        
        self.weights = np.ones(num_features)


    def fit(self, x, y, verbose):
        X_with_bias = x.copy()
        X_with_bias.insert(0, 'bias', 1)
        
        X_matrix = X_with_bias.values
        y_vector = y.values
        n_features = X_matrix.shape[1]
        print(X_matrix.T)
        self.weights = np.ones(n_features)

        for iteration in range(1, self.n_iter+1, 1):
            predictions = np.dot(X_matrix, self.weights)
            errors = predictions - y_vector
            gradient = (2 / len(y_vector)) * (X_matrix.T @ errors)
            self.weights -= self.learning_rate * gradient
            mse = np.mean(errors ** 2)

            if verbose and iteration % verbose == 0:
                print(f'{iteration} | loss: {mse} ')


    def get_coef(self):
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.weights[1:]

    def predict(self, features):
        features_with_bias = features.copy()
        features_with_bias.insert(0, 'bias', 1)
        matrix = features_with_bias.values
        result = np.dot(matrix, self.weights)
        total_sum = np.sum(result)
        return total_sum



lineReg = MyLineReg(50, 0.1)
X = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9]})

y = pd.Series(X['X'] * 2)
lineReg.fit(X, y, False)

df_pd = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9]})
print(df_pd)
# print(lineReg.get_coef())
lineReg.predict(df_pd)
