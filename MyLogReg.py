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

    def calc_roc_auc(self, y_true, y_proba):
        """Вычисление ROC-AUC вручную"""
        if y_proba is None:
            raise ValueError("Для ROC-AUC нужны вероятности, а не классы")
        
        # Сортируем по убыванию вероятности
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_proba_sorted = y_proba[sorted_indices]
        
        # Считаем количество положительных и отрицательных примеров
        n_positive = np.sum(y_true == 1)
        n_negative = np.sum(y_true == 0)
        
        if n_positive == 0 or n_negative == 0:
            return 0.5  # Если нет одного из классов, AUC = 0.5 (случайный классификатор)
        
        # Вычисляем TPR и FPR для разных порогов
        tpr = [0.0]  # True Positive Rate
        fpr = [0.0]  # False Positive Rate
        
        current_tp = 0
        current_fp = 0
        
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                current_tp += 1
            else:
                current_fp += 1
            
            tpr.append(current_tp / n_positive)
            fpr.append(current_fp / n_negative)
        
        # Вычисляем AUC методом трапеций
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        return auc

    def calc_metric(self, y_true, y_predict, y_proba=None):
        def precision():
            tp = np.sum((y_predict == 1) & (y_true == 1))
            tf = np.sum((y_predict == 1) & (y_true == 0))

            if tp + tf == 0:
                return 0.0

            return tp / (tp + tf)
        def accuracy():
            correct_predict = np.sum(y_true == y_predict)
            total = len(y_predict)
            return correct_predict / total

        def recall():
            tp = np.sum((y_predict == 1) & (y_true == 1))
            fn = np.sum((y_predict == 0) & (y_true == 1))

            if tp + fn == 0:
                return 0.0

            return tp / (tp+fn)

        if (self.metric == 'accuracy'):
            return accuracy()
        elif (self.metric == 'precision'):
            return precision()
        elif (self.metric == 'recall'):
            return recall()

        elif (self.metric == 'f1'):
            precision = precision()
            recall = recall()
            
            if (precision + recall) == 0:
                return 0.0

            return 2 * precision * recall / (precision + recall)
        elif (self.metric == 'roc_auc'):
            return self.calc_roc_auc(y_true, y_proba)
        
    def create_matrix(self, data):
        data_with_bias = data.copy()
        data_with_bias.insert(0, 'bias', 1)
        return data_with_bias.values

    def fit(self, X, y, verbose):
        X_matrix = self.create_matrix(X)
        y_vector = y.values
        self.X = X
        self.y_vector = y_vector

        n_samples = X_matrix.shape[0]
        n_features = X_matrix.shape[1]

        self.weights = np.ones(n_features)
       
        eps = 1e-15

        predict = self.sigmoid(X_matrix @ self.weights)
        log_loss = -1*np.mean( y_vector * np.log(predict + eps) + (1-y_vector)*np.log(1-predict + eps) )

        if verbose:
            if self.metric is not None:
                print(f'start | loss: {log_loss:.2f} | {self.metric}:{self.calc_metric(y_vector, self.predict(X), self.predict_proba(X) )}')
            else:
                print(f'start | loss: {log_loss:.2f}')

        for rate in range(self.n_iter):
            predict = self.sigmoid(X_matrix @ self.weights)
            
            log_loss = -1*np.mean( y_vector * np.log(predict + eps) + (1-y_vector)*np.log(1-predict + eps) )
            gradient = (X_matrix.T @ (predict - y_vector)) / n_samples
            self.weights -= self.learning_rate * gradient
            if verbose and rate % verbose == 0:
                if self.metric is not None:
                    print(f'start | loss: {log_loss:.2f} | {self.metric}:{self.calc_metric(y_vector, self.predict(X), self.predict_proba(X) )}')
                else:
                    print(f'start | loss: {log_loss:.2f}')
    
    def get_coef(self):
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.weights[1:]

    def predict_proba(self, X):
        X_matrix = self.create_matrix(X)

        return self.sigmoid(X_matrix @ self.weights)

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def get_best_score(self):
        return self.calc_metric(self.y_vector, self.predict(self.X), self.predict_proba(X))






X = pd.DataFrame({'X': [0,1,2,3,4,5,6,7,8,9,10]})
y = pd.Series([0,1,2,3,4,5,6,7,8,9,10])

log_reg = MyLogReg()
log_reg.fit(X, y, True)



