import numpy as np
import pandas as pd
import random

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate = 0.1, weights = None, metric = None, reg = None, l1_coef = 0, l2_coef = 0, sgd_sample = None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

        random.seed(random_state) 
    
    def __str__(self):
	    return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def _initialize_weights(self, x):
        num_features = len(x)
        
        self.weights = np.ones(num_features)

    def calc_metric(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if (self.metric == 'mae'):
            return np.mean(abs(y_pred - y_true))
        elif self.metric == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_true) ** 2))
        elif self.metric == 'mape':
            epsilon = 1e-15
            return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        elif self.metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-15))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _calculate_regularization_loss(self):
        """Calculate regularization term for the loss function"""
        if self.reg is None:
            return 0
        
        weights_without_bias = self.weights[1:]
        
        if self.reg == 'l1':
            return self.l1_coef * np.sum(np.abs(weights_without_bias))
        elif self.reg == 'l2':
            return self.l2_coef * np.sum(weights_without_bias ** 2)
        elif self.reg == 'elasticnet':
            l1_term = self.l1_coef * np.sum(np.abs(weights_without_bias))
            l2_term = self.l2_coef * np.sum(weights_without_bias ** 2)
            return l1_term + l2_term
        return 0

    def _calculate_regularization_gradient(self):
        """Calculate regularization term for the gradient"""
        if self.reg is None:
            return np.zeros_like(self.weights)
        
        reg_gradient = np.zeros_like(self.weights)
        weights_without_bias = self.weights[1:]
        
        if self.reg == 'l1':
            reg_gradient = self.l1_coef * np.sign(weights_without_bias)
        elif self.reg == 'l2':
            reg_gradient = 2 * self.l2_coef * weights_without_bias
        elif self.reg == 'elasticnet':
            l1_grad = self.l1_coef * np.sign(weights_without_bias)
            l2_grad = 2 * self.l2_coef * weights_without_bias
            reg_gradient[1:] = l1_grad + l2_grad
        
        return reg_gradient

    def fit(self, x, y, verbose):
        sample_number = self.sgd_sample
        
        X_with_bias = x.copy()
        X_with_bias.insert(0, 'bias', 1)
        
        n_features = X_with_bias.shape[1]
        self.weights = np.ones(n_features)
        prev_val = 0
        for iteration in range(1, self.n_iter+1, 1):
            if (sample_number == None):
                X_matrix = X_with_bias.values
                y_vector = y.values
            else:
                if (sample_number < 1):
                    sample_number = int(x.shape[0] * sample_number)
                sample_rows_idx = random.sample(range(x.shape[0]), sample_number)
            
                X_matrix = X_with_bias.iloc[sample_rows_idx].values
                y_vector = y.iloc[sample_rows_idx].values


            predictions = np.dot(X_matrix, self.weights)
            errors = predictions - y_vector
            gradient = (2 / len(y_vector)) * (X_matrix.T @ errors)

            reg_gradient = self._calculate_regularization_gradient()

            total_gradient = gradient + reg_gradient
            
            if (isinstance(self.learning_rate, float)):
                rate = self.learning_rate
            else:
                rate = self.learning_rate(iteration)

            self.weights -= rate * total_gradient
            mse = np.mean(errors ** 2)

            reg_loss = self._calculate_regularization_loss()
            total_loss = mse + reg_loss

            metric_value = None
            if self.metric is not None:
                metric_value = self.calc_metric(y_vector, predictions)

            if verbose and iteration % verbose == 0:
                if self.metric is not None:
                    print(f'{iteration} | loss: {total_loss:.2f} | {self.metric}: {metric_value:.2f}')
                else:
                    print(f'{iteration} | loss: {total_loss:.2f}')
        if self.metric is not None:
            final_predictions = np.dot(X_matrix, self.weights)
            self.best_score = self.calc_metric(y_vector, final_predictions)


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
    
    def get_best_score(self):
        return self.best_score


lineReg = MyLineReg(50, lambda iter: 0.5 * (0.85 ** iter), None, 'mae', 0, 0, 0, 0.5 )
X = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9]})

y = pd.Series(X['X'] * 2)
lineReg.fit(X, y, False)

df_pd = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9]})

# print(lineReg.get_coef())
lineReg.predict(df_pd)
