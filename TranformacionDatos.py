#%%
import pandas as pd
import numpy as np
import seaborn as sb



#%%
url='https://raw.githubusercontent.com/jorgehsaavedra/20201124-test-convocatoria/main/01.%20transformacion_datos/data_fnl.csv'
df=pd.read_csv(url)
ColumnaStren=df.iloc[:,7]
Paradero=df.iloc[:,4]
StrRutaSae=df.iloc[:,11]
Cenefa=df.iloc[:,12]


#%%
#--------------------------------organización de columnas ------------------------------------------------------------

url='https://raw.githubusercontent.com/jorgehsaavedra/20201124-test-convocatoria/main/01.%20transformacion_datos/data_org.csv'
df=pd.read_csv(url, sep=';',encoding='latin-1')
#%%
#--------------------------------Agregar columnas ------------------------------------------------------------

df=df.assign(Paradero=Paradero.values)
df=df.assign(StrRutaSae=StrRutaSae.values)
df=df.assign(Cenefa=Cenefa.values)





#%%
#--------------------------------quitar columnas ------------------------------------------------------------

df.drop(columns='Tipo Vehiculo',inplace=True)


#%%
#-----------------transformar los nombres de las columnas----------------------------------------------------------------

titulos=list(df.columns)


titulos[4],titulos[5],titulos[6],titulos[7],titulos[8],titulos[9],titulos[10]=titulos[10],titulos[6],titulos[9],titulos[5],titulos[7],titulos[8],titulos[4]
df=df[titulos]


df.columns=['FechaContable', 'FechaTransaccion', 'HoraTransaccion', 'Empresa','Paradero', 'StrParadero', 'Linea', 'StrLinea', 'Bus', 'TipoValidacion','RutaSae', 'StrRutaSae', 'Cenefa']



#%%
#--------------------Cambiar formato de fecha ----------------------------------------------------------------------------
df['FechaContable']=pd.to_datetime(df['FechaContable'])
df['FechaTransaccion']=pd.to_datetime(df['FechaTransaccion'])



#%%
#----------------------Extrer datos númericos de una columna StrParadero-----------------------------------------------------

df2=df['StrParadero'].str.extract("(\d*\.?\d+)", expand=True)

df['StrLinea'] = ColumnaStren.values

print(df.info())




