"""
@author : Anish Lakkapragada

Hosts the class, plus a test of the model class inside.
"""
import numpy as np

X = np.random.randn(1000, 2) * 100
X = X/np.max(X)
# correct weights : [1, 1, 1]

y = np.array([[14],
       [29],
       [50],
       [ 2]]).flatten()

def predict_single(x, w, l, a) :
    #print("l * np.power(w, x) : ", l * np.power(w, x))
    w, l = np.array(w), np.array(l)
    return np.sum(l * np.power(w.astype('complex'), x.astype('complex')).astype('float')) + a


def r2_score(y_pred, y_test):
    num = np.sum(np.power(y_test - y_pred, 2))
    denum = np.sum(np.power(y_test - np.mean(y_test), 2))
    return 1 - num / denum


y = np.apply_along_axis(predict_single, 1, X, w = [1, 2], l = [2, 2], a = 1)
y = y / np.max(y)

print(y.shape)

class MultivariateExponentialRegression() :
    def __init__(self, learning_rate = 0.0001):
        self.learning_rate = learning_rate

    '''def fit(self, x_train, y_train, epochs = 10000) :
        m = len(x_train)
        n_features = len(x_train[0])
        weights = np.random.randn(n_features) * np.sqrt(1/3 * n_features)
        lambda_matrix = np.random.randn(n_features) * np.sqrt(1/3 * n_features)
        lr = self.learning_rate
        momentum_w = 0.5
        velocity_w = np.zeros(len(weights))
        print("init weights : ", weights)
        print("init lambda :", lambda_matrix)
        for _ in range(epochs):
            y_hat = np.apply_along_axis(predict_single, 1, x_train, w=weights, l = lambda_matrix)
            #print("This is y_hat : ", y_hat)
            dJdy = (y_hat - y_train) / m
            print("Cost : ", np.sum(np.abs(dJdy)))
            if np.isnan(np.sum(np.abs(dJdy))) :
                break
            dYdtheta = []
            for feature in range(len(weights)):
                X_feature = x_train[:, feature]
                dYdthetaj = lambda_matrix[feature] * X_feature *  np.power((np.zeros(len(X_feature)) + weights[feature]).astype('complex'), (X_feature - 1).astype('complex')).astype('float')
                dYdtheta.append(dYdthetaj)

            dYdtheta = np.array(dYdtheta)

            velocity_w = momentum_w * velocity_w + lr * np.dot(dYdtheta, dJdy)
            weights -= velocity_w
            print("Upated weights : ", weights)
            print("weight grad : ", np.mean(np.abs(np.dot(dYdtheta, dJdy))))
            dYdtheta = []
            for feature in range(len(weights)) :
                X_feature = x_train[:, feature]
                dYdtheta.append(np.power((np.zeros(len(x_train)) + weights[feature]).astype('complex'), X_feature.astype('complex')).astype('float'))
            dYdtheta = np.array(dYdtheta)

            print("DJdy shape : ", dJdy.shape(1, len(dJdy)))
            lambda_matrix -= lr * np.dot(dJdy.reshape(1, len(dJdy)), dYdtheta.T)
            print("updated lambda matrix : ", lambda_matrix)
            print("lambda grad : ", np.mean(np.abs(np.dot(dYdtheta, dJdy))))

        self.weights = weights
        self.lambda_matrix = lambda_matrix'''

    def fit(self, x_train, y_train, epochs = 2000):
        n_features = len(x_train[0])
        weights = np.random.randn(n_features) * np.sqrt(1 /n_features)
        lambda_matrix = np.random.randn(n_features) * np.sqrt(1 /n_features)
        alpha = np.random.randn()
        losses = []
        m = len(y_train)
        lr = self.learning_rate

        for _ in range(epochs):
            y_hat = np.apply_along_axis(predict_single, 1, x_train, w=weights, l=lambda_matrix, a = alpha)
            dJdY_hat = (y_hat - y_train) / m  # shape (N, 1)
            print("Cost : ", np.sum(np.abs(dJdY_hat)))
            losses.append(np.sum(np.abs(dJdY_hat)))
            dYhdTheta = []  # matrix needs to be transposed
            for feature in range(n_features):
                X_feature = x_train[:, feature]
                theta_clone_vector = (np.zeros(len(X_feature)) + weights[feature]).astype('complex')
                dYhdThetaj = lambda_matrix[feature] * X_feature * np.power(theta_clone_vector,
                                                                           (X_feature - 1).astype('complex')).astype(
                    'float')
                dYhdTheta.append(dYhdThetaj)
            dYhdTheta = np.array(dYhdTheta).T  # (N, 3) matrix
            weights -= lr * np.dot(dJdY_hat.T, dYhdTheta).T

            # time to change the lambda matrix too
            dYhdLambda = []
            for feature in range(n_features):
                X_feature = x_train[:, feature]
                theta_clone_vector = (np.zeros(len(X_feature)) + weights[feature]).astype('complex')
                dYhdLambdaj =  np.power(theta_clone_vector, (X_feature).astype('complex')).astype('float')
                dYhdLambda.append(dYhdLambdaj)

            dYhdLambda = np.array(dYhdLambda).T  # (N, 1)
            lambda_matrix -= lr * np.dot(dJdY_hat.T, dYhdLambda).T
            alpha -= lr * np.sum(dJdY_hat)
        self.weights = weights
        self.lambda_matrix = lambda_matrix
        self.alpha = alpha
        self.losses = losses
    def predict(self, x_test) :
        return np.apply_along_axis(predict_single, 1, x_test, w = self.weights, l = self.lambda_matrix, a = self.alpha)
    def evaluate(self, x_test, y_test):
        y_pred = MultivariateExponentialRegression.predict(self, x_test)
        avg_error = np.mean(np.abs(y_pred - y_test))
        return 1 - avg_error/np.mean(y_test)

mexp = MultivariateExponentialRegression(learning_rate = 0.1)
mexp.fit(X[10:], y[10:], epochs = 3000)
print("Evaluation : ", mexp.evaluate(X[:10], y[:10]))