#%%


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.linear_model.logistic

from sklearn.metrics import mean_squared_error, r2_score
from pylab import *

sns.set_style('darkgrid')

#%% 
url='https://raw.githubusercontent.com/jorgehsaavedra/20201124-test-convocatoria/main/02.%20modelacion_datos/data_modelo.csv'
df=pd.read_csv(url)



df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[0:32,0]
KmTT1=df.iloc[0:32,1]
GalT1=df.iloc[0:32,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T1---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))


#%%


######################################################################################################################################################
#----------------------------------------------------Modelo para el el segundo vehiculo---------------------------------------------------------------
df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[32:64,0]
KmTT1=df.iloc[32:64,1]
GalT1=df.iloc[32:64,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T2---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))


#%%

#------------------------------------------------############## Prediccion T3 #############---------------------------------------------------------
####################################################################################################################################################

df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[64:96,0]
KmTT1=df.iloc[64:96,1]
GalT1=df.iloc[64:96,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T3---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))


#------------------------------------------------############## Prediccion T4 #############---------------------------------------------------------
####################################################################################################################################################

df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[96:128,0]
KmTT1=df.iloc[96:128,1]
GalT1=df.iloc[96:128,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T4---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))



#------------------------------------------------############## Prediccion T5 #############---------------------------------------------------------
####################################################################################################################################################

df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[128:160,0]
KmTT1=df.iloc[128:160,1]
GalT1=df.iloc[128:160,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T5---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))
    



#------------------------------------------------############## Prediccion T6 #############---------------------------------------------------------
####################################################################################################################################################

df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[160:192,0]
KmTT1=df.iloc[160:192,1]
GalT1=df.iloc[160:192,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T6---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))
    


#------------------------------------------------############## Prediccion T7 #############---------------------------------------------------------
####################################################################################################################################################

df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[192:224,0]
KmTT1=df.iloc[192:224,1]
GalT1=df.iloc[192:224,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T7---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))

#------------------------------------------------############## Prediccion T8 #############---------------------------------------------------------
####################################################################################################################################################


df2=df[['Pas', 'KmT','Gal']]
df=df2
PasT1=df.iloc[224:260,0]
KmTT1=df.iloc[224:260,1]
GalT1=df.iloc[224:260,2]


Meses=range(0,32)
Meses=list(Meses)

df = pd.DataFrame({'PeriodoAnalizado':Meses,'Pasajeros movilizados':PasT1 , 'Kilometros.T1': KmTT1,'GalT1':GalT1 })

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

Meses=df['Meses'].values
PasT1=df['PasT1'].values
KmTT1=df['KmTT1'].values
GalT1=df['GalT1'].values


X=np.array([Meses,PasT1,KmTT1]).T
Y=np.array(GalT1)

reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(Y,Y_pred))
r2=reg.score(X,Y)

print("El error es")
print("El valor de r cuadrado es :",r2)
print("los coeficientes son:\n'",reg.coef_)


Meses=32
PasT1=52450961
KmTT1=285347

#%%

#--------------------------------- Pediccion de consumo durante los proximos 12 meses para el T8---------------------------------------------------
 
for i in range(0,13):
 
    print("Prediccion de consumo de conbustible: \n",reg.predict([[Meses+i,PasT1,KmTT1]]))