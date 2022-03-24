#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display
import scipy.stats as stats


##Normal distribution for iris dataset
def Gaussian_plot(DB, shared_variance=False):
    '''
    Density plot of P(petal_width|y=c) assuming Gaussin pdfs for each class
    '''
    
    iris = sns.load_dataset('iris')
    iris = iris.loc[iris.species != 'setosa', :]
    plt.figure(figsize=(15,7))
    
    mu1 = iris.groupby('species')['petal_width'].mean()['virginica']
    std1 = iris.groupby('species')['petal_width'].std()['virginica']

    mu2 = iris.groupby('species')['petal_width'].mean()['versicolor']
    std2 = iris.groupby('species')['petal_width'].std()['versicolor']
    
    if shared_variance:
        std =np.sqrt( 0.5*std2**2 +  0.5*std1**2 )
        std1 = std
        std2 = std

    x1 = np.linspace(mu1 - 3*std1, mu1 + 3*std1, 100)
    plt.fill_between(x1, stats.norm.pdf(x1, mu1, std1), alpha=0.5, label="virginica")

    x2 = np.linspace(mu2 - 3*std2, mu1 + 3*std2, 100)
    plt.fill_between(x2, stats.norm.pdf(x2, mu2, std2), alpha=0.5, label="versicolor")
    plt.vlines(DB,0,2.1)
    plt.text(0.9,1.5, r'$P(X|C_1)$', fontsize=20)
    plt.text(2.1,1.7, r'$P(X|C_2)$', fontsize=20)
    plt.xlabel("petal_width")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
