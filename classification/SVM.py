#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import load_diabetes
from sklearn.model_selection import  train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import PolynomialFeatures


##plot desicion region SVM
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=200, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

 
#Diferent basis function
def kernel_trick(basis_function='poly'):
    x = np.linspace(-4,4, 20)
    y = ((x < 2) & (x > -2))*1
    z = np.zeros(shape=20)
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    if basis_function == 'poly':
        gamma = 1.0
        r=0.1
        x_t = (gamma*x + r)**2
        ax[1].set_title("Polynomial", fontsize=15)
        ax[0].set_title("Original", fontsize=15)
        ax[0].scatter(x, z, c=y, cmap='Set1')
        ax[1].scatter(x, x_t, c=y, cmap='Set1')
        ax[0].set_xlabel('x', fontsize=15)
        ax[1].set_xlabel('x', fontsize=15)
        ax[1].set_ylabel(r'$(\gamma x + r)^2$', fontsize=15)

    elif basis_function == 'gaussian rbf':
        gamma = 0.1
        x_t1 = np.exp(-gamma*np.abs(x-1)**2)
        x_t2 = np.exp(-gamma*np.abs(x-3)**2)
        
        ax[0].scatter(x, z, c=y, cmap='Set1')
        ax[1].scatter(x_t2, x_t1, c=y, cmap='Set1')
        ax[1].set_title("Gaussian RBF", fontsize=15)
        ax[0].set_title("Original", fontsize=15)
        ax[0].set_xlabel('x', fontsize=15)
        ax[1].set_xlabel(r'$\gamma (x-\mu_2)^2$', fontsize=15)
        ax[1].set_ylabel(r'$\gamma (x-\mu_1)^2$', fontsize=15)

    elif basis_function == 'sigmoid':
        gamma = 1.0
        sigmoid = lambda x:1/(1+np.exp(-x))
        x_t1 = sigmoid(gamma*(x-1)+ 2)
        x_t2 = sigmoid(gamma*(x-4)+ 2)
        ax[0].scatter(x, z, c=y, cmap='Set1')
        ax[1].scatter(x_t1, x_t2, c=y, cmap='Set1')
        ax[1].set_title("sigmoid", fontsize=15)
        ax[0].set_title("Original", fontsize=15)
        ax[0].set_xlabel('x', fontsize=15)
        ax[1].set_xlabel(r'$\sigma (\gamma (x-\mu_2) + r)$', fontsize=15)
        ax[1].set_ylabel(r'$\sigma (\gamma (x-\mu_1) + r)$', fontsize=15)
