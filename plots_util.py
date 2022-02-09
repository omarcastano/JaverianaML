from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Plot several boxplots using seaborn
def seabornboxplot(data: pd.DataFrame, y: list):
  '''
  Plot boxplots using seaborn
  
  args:
    data: data frame
    y: name of variables to plot 
  '''
    
    fig , ax = plt.subplots(ncols=len(y), figsize=(30,7))
    for i, col in enumerate(y):
        sns.boxplot(data=data, y=col, ax=ax[i])
        
#Plot several boxplots using plotly
def plotlyboxplot(data, y):
    '''
    Plot boxplots using plotly

    args:
      data: data frame
      y: name of variables to plot 
    '''
  
    fig = make_subplots(rows=1, cols=len(y))

    for i, col in enumerate(y):
        fig.add_trace(go.Box(y=dataset[col], name=col), row=1, col=i+1 )

    fig.show()
