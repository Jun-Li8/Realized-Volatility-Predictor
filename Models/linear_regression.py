import numpy as np

class LinearRegression():
    def __init__(self):
        self.weights = None

    #Uses close-form solution: w = ((X^T*X)^(-1))*X^T*y
    def fit(self, features, targets):
        ones = np.ones((len(features),1))
        features_new = np.concatenate((ones,features),axis = 1)
        transpose_x = np.transpose(features_new)
        XT_X_inv = np.linalg.inv(np.matmul(transpose_x,features_new))
        XT_X_Inv_XT = np.matmul(XT_X_inv,transpose_x)
        self.weights = np.matmul(XT_X_Inv_XT, targets)

    #Apply weights to the independent variables
    def predict(self,features):
        print(self.weights)
        ones = np.ones((len(features),1))
        features_new = np.concatenate((ones,features),axis = 1)
        return np.matmul(features_new,self.weights)

    #Return the coefficient of determination R^2 of the prediction
    def score(self, features, targets):
        prediction = self.predict(features)
        u = np.sum((targets-prediction)**2)
        v = np.sum((targets-np.mean(targets))**2)
        return (1-(u/v))
