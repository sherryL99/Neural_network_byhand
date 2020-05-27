import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm
from cvxopt import solvers, matrix


class SVM:
    def __init__(self):
        self.X1 = self.get_random_points(0, 1, 100)
        self.X2 = self.get_random_points(0, 1, 100)
        self.X = [(ele1,ele2) for ele1,ele2 in zip(self.X1,self.X2)]
        self.D = self.get_desired_points()
    # Updated the previous function to randomly assign randomNeural_Network
    # points of any length with the specified range
    def get_random_points(self, a, b, n):
        x = list()
        for i in range(n):
            temp = random.uniform(a, b)
            x.append(temp)
        return x

    def get_desired_points(self):
        d = list()
        for x1,x2 in self.X:
            if (x2 < 1/5 * math.sin(10*x1) + 0.3) or ((x2 - 0.8)**2 + (x1 - 0.5)**2 < (0.15)**2):
                d.append(1)
            else:
                d.append(-1)
        return d

    def linear_kernel(self,xi,xj):
        return np.dot(xi,xj)

    def polynomial_kernel(self,Xi,Xj,d):
        return (1 + np.dot(Xi.T,Xj))**d

    def gaussian_kernel(self,Xi, Xj, sigma):
        return math.exp(-(norm(Xi - Xj) ** 2 / sigma ** 2))

    def get_p(self,X, D, kernel='poly', param=2):
        P = []
        for i in range(len(D)):
            row = []
            for j in range(len(D))  :
                if kernel is 'poly':
                    temp = D[i] * D[j] * self.polynomial_kernel(np.array(X[i]), np.array(X[j]), param)
                elif kernel is 'gaussian':
                    temp = D[i] * D[j] * self.gaussian_kernel(np.array(X[i]), np.array(X[j]), param)
                elif kernel is 'linear':
                    temp = D[i] * D[j] * self.linear_kernel(np.array(X[i]), np.array(X[j]))
                row.append(temp)
            P.append(row)

    def get_alpha(self,X,D,kernel='poly',param=2):
        P = self.get_p(X,D,kernel,param)
        q = matrix(-1 * np.ones(len(D)))
        h = matrix(np.zeros(len(D)))
        G = matrix(-1 * np.eye(len(D)))
        b = matrix([0], (1, 1), 'd')
        A = matrix(D, (1, len(D)), 'd')
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(list(sol['x']))
        return alpha

    def get_support_vector(self,alpha):
        k = alpha.argsort()[::-1]
        return k[0]

    def get_theta(self, alpha, X, D, K, kernel='poly', param=2):
        temp = 0
        for i in range(len(D)):
            if kernel is 'poly':
                temp += - alpha[i] * D[i] * self.polynomial_kernel(np.array(X[i]), np.array(X[K]),
                                                              param)  # param: degree for poly kernel
            elif kernel is 'linear':
                temp += - alpha[i] * D[i] * self.linear_kernel(np.array(X[i]), np.array(X[K]))
            elif kernel is 'gaussian':
                temp += - alpha[i] * D[i] * self.gaussian_kernel(np.array(X[i]), np.array(X[K]), param)  # parma: sigma
        theta = D[K] - temp
        return theta

    def get_gx(self,alpha, X, D, theta, Xk, kernel='poly', param=2):
        temp = 0
        for i in range(len(D)):
            if kernel is 'poly':
                temp += alpha[i] * D[i] * self.polynomial_kernel(np.array(X[i]), Xk, param)
            elif kernel is 'linear':
                temp += alpha[i] * D[i] * self.linear_kernel(np.array(X[i]), Xk)
            elif kernel is 'gaussian':
                temp += alpha[i] * D[i] * self.gaussian_kernel(np.array(X[i]), Xk, param)
        g_x = temp + theta
        return (g_x)

if __name__ == '__main__':
    ob = SVM()
    X = ob.X
    D = ob.D
    alpha = ob.get_alpha(X,D,'poly',2)
    K = ob.get_support_vector(alpha)
    for (x, y), d in zip(X, D):
        if d is 1:
            plt.scatter(x, y, color='b', marker='o')
        elif d is -1:
            plt.scatter(x, y, color='r', marker='o')
    new_range = np.arange(0, 1, 0.1)
    theta = ob.get_theta(alpha,X, D, K, 'poly', 2)
    for x1 in new_range:
        for x2 in new_range:
            g = ob.get_gx(alpha, X, D, theta, np.array([x1, x2]), 'poly', 2)
            print(g)
            if (g <= 1.1 and g >= 0.9):
                plt.scatter(x1, x2, color='m', marker='d')
            if ((g >= -1.1) and (g <= -0.9)):
                plt.scatter(x1, x2, color='c', marker='d')
            if ((g >= -0.1) and (g <= 0.1)):
                plt.scatter(x1, x2, color='y', marker='d')
    plt.show()


