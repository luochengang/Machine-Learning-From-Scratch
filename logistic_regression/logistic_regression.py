# -*- coding:utf-8 -*-
from sklearn import datasets
import numpy as np
from phe import paillier

# Import helper functions
from utils import make_diagonal, normalize, train_cv_test_split, accuracy_score
from utils import Plot

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
            梯度下降的轮数
        learning_rate: float
            梯度下降学习率
            We recommend trying values of the learning rate α on a log-scale, at multiplicative
            steps of about 3 times the previous value (i.e., 0.3, 0.1, 0.03, 0.01, 0.003, 0.001 and so on).
        la: float
            正则化项系数lambda
            regularization parameter lambda一般从0、0.01开始尝试，每次乘以2，一直到10.24
    """
    def __init__(self, learning_rate=0.1, la=0.16, n_iterations=4000):
        self.learning_rate = learning_rate
        self.la = la
        self.n_iterations = n_iterations

    def initialize_weights(self, na_features, nb_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / na_features)
        wa = np.random.uniform(-limit, limit, (na_features, 1))
        wb = np.random.uniform(-limit, limit, (nb_features, 1))
        b = 0
        # np.insert() https://blog.csdn.net/Mxeron/article/details/113405004
        # axis=0 按行插入
        # axis=1 按列插入
        self.wa = np.insert(wa, 0, b, axis=0)
        self.wb = wb

    def fit(self, XA, XB, y):
        m_samples, na_features = XA.shape
        m_samples, nb_features = XB.shape
        '''
        未完成 v数组
        v = np.random.uniform(1, 2, nb_features)
        i = 0
        while i < nb_features:
            XB[:, i] /= v[i]
            i += 1
        '''

        self.initialize_weights(na_features, nb_features)
        # 为X增加一列特征x1，x1 = 0
        XA = np.insert(XA, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 生成公私钥
        public_key, private_key = paillier.generate_paillier_keypair()

        # 梯度训练n_iterations轮
        for i in range(self.n_iterations):
            # X.dot() 矩阵乘积
            h_x = XA.dot(self.wa) + XB.dot(self.wb)
            y_pred = sigmoid(h_x)

            inside = y_pred - y
            encry_inside = np.array([], dtype=paillier.EncryptedNumber)
            for i, x in np.ndenumerate(inside):
                encry_inside = np.append(encry_inside, public_key.encrypt(x))
            encry_inside = encry_inside.reshape(inside.shape)

            wa_grad = XA.T.dot(inside) / m_samples + self.la / m_samples * self.wa
            self.wa = self.wa - self.learning_rate * wa_grad

            encry_wb_grad = XB.T.dot(encry_inside) / m_samples + self.la / m_samples * self.wb
            rb = np.random.uniform(-10, 10, encry_wb_grad.shape)
            encry_wb_grad_add = encry_wb_grad + rb

            wb_grad_add = np.zeros(encry_wb_grad.shape)
            for i, x in np.ndenumerate(encry_wb_grad_add):
                wb_grad_add[i] = private_key.decrypt(x)

            wb_grad = wb_grad_add - rb
            self.wb = self.wb - self.learning_rate * wb_grad


    def predict(self, XA, XB):
        XA = np.insert(XA, 0, 1, axis=1)
        h_x = XA.dot(self.wa) + XB.dot(self.wb)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)




def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    XA_train, XB_train, XA_cv, XB_cv, XA_test, XB_test, y_train, y_cv, y_test = train_cv_test_split(X, y, seed=1)

    clf = LogisticRegression(learning_rate=0.1, la=0.16, n_iterations=4)
    clf.fit(XA_train, XB_train, y_train)
    y_pred = clf.predict(XA_cv, XB_cv)
    y_pred = np.reshape(y_pred, y_cv.shape)

    accuracy = accuracy_score(y_cv, y_pred)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
