# -*- coding:utf-8 -*-
from sklearn import datasets
import numpy as np
from phe import paillier
import os
import sys
from ucimlrepo import fetch_ucirepo

from sklearn.utils.multiclass import type_of_target
'''
python命令行运行找不到自定义模块
https://blog.csdn.net/qq_40472613/article/details/119670598
'''
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import helper functions
from utils import make_diagonal, normalize, train_test_split, train_cv_test_split, train_cv_test_split_fed, accuracy_score
from utils import Plot
import pymysql
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.metrics import average_precision_score

conn = pymysql.connect(host="1.94.57.226", user="root", password="QY9cfB3hhK", database="ry-vue")
cursor = conn.cursor()

def sigmoid(x):
    x = np.array(x, dtype=np.float64)
    # return 1 / (1 + np.exp(-x))
    '''
    Python中sigmoid函数中报： RuntimeWarning: overflow encountered in exp
    https://blog.csdn.net/guyu1003/article/details/108470391
    '''
    return expit(x)


class LogisticRegressionFed():
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

            wa_reg = np.insert(self.wa[1:, :], 0, 0, axis=0)
            wa_grad = XA.T.dot(inside) / m_samples + self.la / m_samples * wa_reg
            self.wa = self.wa - self.learning_rate * wa_grad

            wb_reg = np.insert(self.wb[1:, :], 0, 0, axis=0)
            encry_wb_grad = XB.T.dot(encry_inside) / m_samples + self.la / m_samples * wb_reg
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
    def __init__(self, learning_rate=0.1, la=0.16, n_iterations=40000):
        self.learning_rate = learning_rate
        self.la = la
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        # np.insert() https://blog.csdn.net/Mxeron/article/details/113405004
        # axis=0 按行插入
        # axis=1 按列插入
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        '''
        未完成 v数组
        v = np.random.uniform(1, 2, nb_features)
        i = 0
        while i < nb_features:
            XB[:, i] /= v[i]
            i += 1
        '''

        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        for i in range(self.n_iterations):
            # X.dot() 矩阵乘积
            h_x = X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_reg = np.insert(self.w[1:, :], 0, 0, axis=0)
            w_grad = X.T.dot(y_pred - y) / m_samples + self.la / m_samples * w_reg
            self.w = self.w - self.learning_rate * w_grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)

    # 用于计算AUC值
    def getScore(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        return sigmoid(h_x)


# 中心化的逻辑回归 特征缩放
class LogisticRegressionScale():
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
    def __init__(self, learning_rate=0.1, la=0.16, n_iterations=40000):
        self.learning_rate = learning_rate
        self.la = la
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        # np.insert() https://blog.csdn.net/Mxeron/article/details/113405004
        # axis=0 按行插入
        # axis=1 按列插入
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape

        # 计算每一列的平均值
        column_means = np.mean(X, axis=0)
        self.column_means = column_means
        # 计算每一列的标准差
        column_stds = np.std(X, axis=0)
        self.column_stds = column_stds
        # 标准化每一列
        X = (X - column_means) / column_stds

        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        for i in range(self.n_iterations):
            # X.dot() 矩阵乘积
            h_x = X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_reg = np.insert(self.w[1:, :], 0, 0, axis=0)
            w_grad = X.T.dot(y_pred - y) / m_samples + self.la / m_samples * w_reg
            self.w = self.w - self.learning_rate * w_grad

    def predict(self, X):
        # 标准化每一列
        X = (X - self.column_means) / self.column_stds
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)

    # 用于计算AUC值
    def getScore(self, X):
        # 标准化每一列
        X = (X - self.column_means) / self.column_stds
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        return sigmoid(h_x)


def testFed(X, y):
    XA_train, XB_train, XA_cv, XB_cv, XA_test, XB_test, y_train, y_cv, y_test = train_cv_test_split_fed(X, y, seed=1)

    clf = LogisticRegressionFed(learning_rate=0.1, la=0.16, n_iterations=4)
    clf.fit(XA_train, XB_train, y_train)
    y_pred = clf.predict(XA_cv, XB_cv)
    y_pred = np.reshape(y_pred, y_cv.shape)

    accuracy = accuracy_score(y_cv, y_pred)
    print("Accuracy:", accuracy)


def test(X, y):
    X_train, X_cv, X_test, y_train, y_cv, y_test = train_cv_test_split(X, y, seed=1)

    # 正则化项系数lambda为0.02较好
    clf = LogisticRegression(learning_rate=0.01, la=0, n_iterations=400000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_cv)
    y_pred = np.reshape(y_pred, y_cv.shape)

    accuracy = accuracy_score(y_cv, y_pred)
    print("Accuracy:", accuracy)

    y_pred = clf.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def noLambdaTest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)

    clf = LogisticRegressionScale(learning_rate=0.01, la=0, n_iterations=40000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    y_scores = clf.getScore(X_test)
    y_scores = np.reshape(y_scores, y_test.shape)
    # print(type_of_target(y_test))
    y_test = y_test.astype(int)
    # print(type_of_target(y_test))
    '''
    roc_auc_score raise ValueError("{0} format is not supported".format(y_type))
    https://blog.csdn.net/ztf312/article/details/101265425
    https://www.cnpython.com/qa/1321258
    '''
    auc_score = roc_auc_score(y_test, y_scores)
    print("AUC score:", auc_score)


def getDefaultOfCreditCardClientsDataset():
    df = pd.read_excel("./DefaultOfCreditCardClients.xlsx")
    # 将DataFrame转换为NumPy数组
    numpy_array = df.to_numpy()
    # print(numpy_array)
    # 提取特征X（所有列除了最后一列）
    X = numpy_array[:, :-1]
    # 提取标签y（最后一列）
    y = numpy_array[:, -1]
    return X, y


def getCreditCardFraudDetection():
    df = pd.read_excel("./creditcard.xlsx")
    # 将DataFrame转换为NumPy数组
    numpy_array = df.to_numpy()
    # print(numpy_array)
    # 提取特征X（所有列除了最后一列）
    X = numpy_array[:, :-1]
    # 提取标签y（最后一列）
    y = numpy_array[:, -1]
    return X, y


def getBreastCancerDataset():
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    X = X.values
    y = breast_cancer_wisconsin_diagnostic.data.targets
    y = y.values
    y[y == "B"] = 0
    y[y == "M"] = 1
    # numpy将二维（多维）数组降为一维数组
    y = np.reshape(y, y.shape[0])

    # metadata
    # print(breast_cancer_wisconsin_diagnostic.metadata)

    # variable information
    # print(breast_cancer_wisconsin_diagnostic.variables)
    return X, y


def getDataFromModule():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    return X, y


def main():
    # Load dataset
    X, y = getBreastCancerDataset()
    noLambdaTest(X, y)


if __name__ == "__main__":
    main()
