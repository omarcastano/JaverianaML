import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

#Implements polinomial regression
def PolynomialRegression(degree, X_train, X_test, y_train, y_test):
    #Pipeline
    model  = Pipeline(steps=[
                         ('poly', PolynomialFeatures(degree=degree)),
                         ('lin_reg', LinearRegression())])

    #Fit the model
    model.fit(X_train, y_train)

    #Plot model
    fig = plt.figure(figsize=(7,6))
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    print('r2_train =', model.score(X_train, y_train))
    print('r2_valid =', model.score(X_test, y_test))
    x = np.linspace(0,1).reshape(-1,1)
    plt.plot(x, model.predict(x), 'r')
    plt.xlim(-0.01, 1)
    plt.ylim(-0.1, 11)