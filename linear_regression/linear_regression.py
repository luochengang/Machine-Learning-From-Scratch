import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing

from utils import train_test_split, train_cv_test_split, calculate_loss
from utils import mean_squared_error, Plot

# L1正则化
class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha

    # L1正则化的方差
    def __call__(self, w):
        loss = np.sum(np.fabs(w))
        return self.alpha * loss

    # L1正则化的梯度
    def grad(self, w):
        return self.alpha * np.sign(w)


# L2正则化
class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha

    # L2正则化的方差
    def __call__(self, w):
        loss = w.T.dot(w)
        return self.alpha * 0.5 * float(loss)

    # L2正则化的梯度
    def grad(self, w):
        return self.alpha * w


class LinearRegression():
    """
    Parameters:
    -----------
    n_iterations: int
        梯度下降的轮数
    learning_rate: float
        梯度下降学习率
    regularization: l1_regularization or l2_regularization or None
        正则化
    gradient: Bool
        是否采用梯度下降法或正规方程法。
        若使用了正则化，暂只支持梯度下降
    """

    def __init__(self, n_iterations=3000, learning_rate=0.01, alpha=0.02):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.scaling_X = None

    def initialize_weights(self, n_features):
        # 初始化参数
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        self.training_errors = []
        # 计算每列的平均值
        self.scaling_X = np.mean(X, axis=0)
        X = X / self.scaling_X

        # 梯度下降
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)

            w_reg = np.insert(self.w[1:, :], 0, 0, axis=0)
            loss = np.mean(0.5 * (y_pred - y) ** 2) + self.alpha / (2 * m_samples) * float(w_reg.T.dot(w_reg))  # 计算loss
            # print(loss)
            self.training_errors.append(loss)
            w_grad = X.T.dot(y_pred - y) / m_samples + self.alpha / m_samples * w_reg  # (y_pred - y).T.dot(X)，计算梯度
            self.w = self.w - self.learning_rate * w_grad  # 更新权值w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        X = X / self.scaling_X
        y_pred = X.dot(self.w)
        return y_pred





def main():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    # m_samples, n_features = np.shape(X)
    # m_samples, n_features = housing.data.shape

    X_train, X_cv, X_test, y_train, y_cv, y_test = train_cv_test_split(X, y, seed=1)

    # 可自行设置模型参数，如正则化，梯度下降轮数学习率等
    model = LinearRegression(n_iterations=40000, learning_rate=0.0001, alpha=0.04)

    model.fit(X_train, y_train)


    # Training error plot 画loss的图
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()


    y_pred = model.predict(X_cv)
    y_pred = np.reshape(y_pred, y_cv.shape)

    loss = calculate_loss(y_cv, y_pred)
    print("loss:", loss)

    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    loss = calculate_loss(y_test, y_pred)
    print("loss:", loss)



    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error: %s" % (mse))

    '''
    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results，画拟合情况的图
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()
    '''


if __name__ == "__main__":
    main()