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
def Binary_Gaussian_Plot(DB, shared_variance=False):
    '''
    Density plot of P(petal_width|y=c) assuming Gaussin pdfs for each class
    '''
    fig, ax = plt.subplots(1,2,figsize=(20,7))
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
    ax[0].fill_between(x1, stats.norm.pdf(x1, mu1, std1)*0.5, alpha=0.5, label="virginica")

    x2 = np.linspace(mu2 - 3*std2, mu2 + 3*std2, 100)
    ax[0].fill_between(x2, stats.norm.pdf(x2, mu2, std2)*0.5, alpha=0.5, label="versicolor")
    sns.scatterplot(data=iris, x='petal_width', y='species', hue='species', hue_order =['virginica', 'versicolor'], ax=ax[1])

    ax[0].vlines(DB,0,1)
    ax[1].vlines(DB,0,1)
    ax[0].text(0.7,0.9, r'$P(X|C_1)P(C_1)$', fontsize=20)
    ax[0].text(1.9,0.9, r'$P(X|C_2)P(C_2)$', fontsize=20)
    ax[0].set_xlabel("petal_width")
    ax[0].set_ylabel("Density")
    ax[0].legend()

def Multiclass_Gaussian_Plot(DB1, DB2, shared_variance=False):
    '''
    Density plot of P(petal_width|y=c) assuming Gaussin pdfs for each class
    '''
    fig, ax = plt.subplots(1,2,figsize=(25,7))
    iris = sns.load_dataset('iris')
    plt.figure(figsize=(25,7))
    
    mu1 = iris.groupby('species')['petal_width'].mean()['virginica']
    std1 = iris.groupby('species')['petal_width'].std()['virginica']

    mu2 = iris.groupby('species')['petal_width'].mean()['versicolor']
    std2 = iris.groupby('species')['petal_width'].std()['versicolor']

    mu3 = iris.groupby('species')['petal_width'].mean()['setosa']
    std3 = iris.groupby('species')['petal_width'].std()['setosa']

    
    if shared_variance:
        std =np.sqrt( (50/150)*std2**2 +  (50/150)*std1**2 + (50/150)*std3**2 )
        std1 = std
        std2 = std
        std3 = std

    x1 = np.linspace(mu1 - 3*std1, mu1 + 3*std1, 100)
    ax[0].fill_between(x1, stats.norm.pdf(x1, mu1, std1)*0.5, alpha=0.5, label="virginica")

    x2 = np.linspace(mu2 - 3*std2, mu2 + 3*std2, 100)
    ax[0].fill_between(x2, stats.norm.pdf(x2, mu2, std2)*0.5, alpha=0.5, label="versicolor")

    x3 = np.linspace(mu3 - 3*std3, mu3 + 3*std3, 100)
    ax[0].fill_between(x3, stats.norm.pdf(x3, mu3, std3)*0.5, alpha=0.5, label="versicolor")

    sns.scatterplot(data=iris, x='petal_width', y='species', hue='species', hue_order =['virginica', 'versicolor', 'setosa'], ax=ax[1])

    ax[0].vlines(DB1,0,1.8)
    ax[1].vlines(DB1,0,2)

    ax[0].vlines(DB2,0,1.8)
    ax[1].vlines(DB2,0,2)


    ax[0].text(0.9,1.0, r'$P(X|C_1)P(C_1)$', fontsize=15)
    ax[0].text(1.9,1.0, r'$P(X|C_2)P(C_2)$', fontsize=15)
    ax[0].text(0.0, 1.0, r'$P(X|C_3)P(C_3)$', fontsize=15)
    ax[0].set_xlabel("petal_width")
    ax[0].set_ylabel("Density")
    ax[0].legend() 


   
def Gaussian_3d_plot(w0, w):

    from plotly.subplots import make_subplots
    import scipy.stats as stats
    import plotly.graph_objects as go
    import plotly.express as px
    iris = sns.load_dataset('iris')
    iris = iris.loc[iris.species != 'setosa', :]

    means = iris.groupby("species", as_index=False).mean()[['species','petal_width', 'petal_length']]
    mu1 = means.loc[means.species=='versicolor', ['petal_width', 'petal_length']].to_numpy().ravel()
    mu2 = means.loc[means.species=='virginica', ['petal_width', 'petal_length']].to_numpy().ravel()

    cov1 = iris.loc[iris.species=='versicolor' ,['petal_width', 'petal_length']].cov().to_numpy()
    cov2 = iris.loc[iris.species=='virginica' ,['petal_width', 'petal_length']].cov().to_numpy()

    cov = 0.5*cov1 + 0.5*cov2

    xs = np.linspace(0.5, 3.0, 100)
    zs = np.linspace(0, 1.4, 100)

    X, Z = np.meshgrid(xs, zs)
    Y = w*X + w0

    x1, y1 = np.mgrid[0.5:1.9:200j, 2.5:5.3:200j]
    x2, y2 = np.mgrid[1.3:2.6:200j, 4.5:7:200j]


    xy1 = np.column_stack([x1.flat, y1.flat])
    xy2 = np.column_stack([x2.flat, y2.flat])

    cmap = plt.get_cmap("tab10")
    colorscale = [[0, 'rgb' + str(cmap(1)[0:3])], [1, 'rgb' + str(cmap(2)[0:3])]]

    z1 = stats.multivariate_normal.pdf(xy1, mean=mu1, cov=cov)
    z1 = z1.reshape(x1.shape)

    z2 = stats.multivariate_normal.pdf(xy2, mean=mu2, cov=cov)
    z2 = z2.reshape(x2.shape)

    fig = make_subplots(rows=1, cols=2, specs=[[{'is_3d': True}, {'is_3d': False}]])
    trace_a = go.Surface(x=x1, y=y1, z=z1, cmin=0, cmax=1, surfacecolor=np.zeros(shape=z1.shape), colorscale= colorscale, showscale=False, name="versicolor", opacity=0.8) 
    trace_b = go.Surface(x=x2, y=y2,z=z2, cmin=0, cmax=1, surfacecolor=np.ones(shape=z2.shape) , colorscale= colorscale, showscale=False, name="virginica", opacity=0.8)
    trace_c = go.Surface(x=X, y=Y,z=Z, cmin=0, cmax=1, surfacecolor=np.ones(shape=z2.shape)*0.5 , colorscale= colorscale, showscale=False, name="virginica")
    fig.add_traces([trace_a, trace_b, trace_c], rows=1, cols=1)
    fig.update_scenes(xaxis_title_text="petal_width", yaxis_title_text="petal_length")

    scatter_1 = go.Scatter(x=iris[iris.species=='versicolor'].petal_width, y=iris[iris.species=='versicolor'].petal_length,  mode='markers', name='versicolor', marker_color='red')
    scatter_2 = go.Scatter(x=iris[iris.species=='virginica'].petal_width, y=iris[iris.species=='virginica'].petal_length,  mode='markers', name='virginica', marker_color='green')

    x=np.linspace(1,2.5,100)
    y=w*x+w0
    line = go.Scatter(x=x, y=y,  mode='lines', name='virginica', marker_color='black')

    fig.add_traces([scatter_1, scatter_2, line], rows=1, cols=2)
    fig.update_yaxes(title_text="petal_length", row=1, col=2)
    fig.update_xaxes(title_text="petal_width", row=1, col=2)
    fig.update_layout(width=900*2, height=600)
    fig.show()
