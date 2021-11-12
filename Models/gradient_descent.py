import numpy as np
from loss import SquaredLoss

class GradientDescent():
    def __init__(self, regularization=None, learning_rate = 0.0000012):
        self.learning_rate = learning_rate
        self.model = None
        self.loss = SquaredLoss(None)

    def fit(self, features, targets, max_iter = 100000):
        #Initialize parameters
        param = (np.array([74.0, 1.0]).reshape(1,-1))

        #Include w_0
        f = np.concatenate((features, np.ones([len(features),1])), axis = 1)

        #Trackering variables
        iter = 0
        prev_loss = 0
        not_converged = True

        #Training
        while not_converged and iter < max_iter:
            if abs(prev_loss - self.loss.loss(f,targets, param)) < 0.0001:
                not_converged = False
            prev_loss = self.loss.loss(f,targets, param)
            grad = self.loss.gradient(f,targets, param)
            param -= self.learning_rate*grad
            iter += 1
        #Model
        self.model = param

    def predict_quantized(self, features):
        unquantized_prediction = predict(features)
        quantized_prediction = np.zeros(len(unquantized_prediction))

        for i, val in enumerate(confidence):
            if val >= .5:
                prediction[i] = 1
            elif val < 0.5:
                prediction[i] = 0

            return prediction


    def predict(self, features):
        f = np.concatenate((features, np.ones([len(features),1])), axis = 1)
        return np.dot(self.model, np.transpose(f))

    def score(self, features, targets):
        print(self.model)
        prediction = self.predict(features)
        u = np.sum((targets-prediction)**2)
        v = np.sum((targets-np.mean(targets))**2)
        return (1-(u/v))
