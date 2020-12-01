import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from pylab import *

#-----------------------------------------------------------Correlación de datos de la visualización-------------------------------------------------

print( "iniciamos la emulación")
url=pd.read_csv('https://raw.githubusercontent.com/jorgehsaavedra/20201124-test-convocatoria/main/03.%20visualizacion_datos/data_viz.csv')
df=url

nuevo=df
corr = nuevo.corr () 
sns.heatmap (
        corr, 
        xticklabels = corr.columns, 
        yticklabels = corr.columns)
print(corr)

m=nuevo.corr(method='spearman')

print(m)

#%% 
sns.set_style('darkgrid')
g=sns.pairplot(nuevo,diag_kind="hist")
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(),rotation=45)
                 
show()   

print(df.info())