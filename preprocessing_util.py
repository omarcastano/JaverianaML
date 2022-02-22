import pandas as pd
import numpy as np

def OutlierDetection(data:pd.DataFrame, variable:str, factor:float = 1.5):

    '''
    Dedect outliers based on Inter Quantile Range

    Args:
        data: DataFrame
        varaible: varaible to detect outliers
        factor: factor to detect outliers usgin the expresions Q3 + factor*IQR  
                and Q3 - factor*IQR (Default factor=1.5)
    '''

    q1 = data[variable].quantile(q=0.25)
    q3 = data[variable].quantile(q=0.75)
    IQR = q3 - q1 

    return data.loc[(data[variable] > q3 + factor*IQR) & (data[variable] < q1 - factor*IQR), variable]
