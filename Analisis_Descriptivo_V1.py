# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:29:32 2020

@author: tomvc
"""

import pandas as pd

#Se hace la extraccion de las 2 bases de Datos a Usar
#Data: Tabla de casos positivos en Colombia INS, Datos.gov.co
#Poblacion: Tabla proyecciones poblacion 2020, Dane
from download_data import data, poblacion 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Se definen las funciones con las columnas de datos que seran usadas posteriormente
#Nota: No debemos traer columnas inutiles, como grupo etnico, etnia, codigo pais, Codigo municipio, codigo departamento.
df_pruebas, df_casos = data(10000000)
df_pob = poblacion('poblacion_municipios.xlsx')
df_pob['Región'] = df_pob['Región'].replace('Bogotá, D.C.', 'Bogotá D.C.')
df_pob['Región'] = df_pob['Región'].replace('Cartagena', 'Cartagena de Indias')
df_casos.columns = ['ID', 'Fecha', 'Cod_Municipio', 'Ciudad', 'Depto', 'Estado', 'Edad', 'Sexo', 'Tipo', 'Gravedad',
                   'Pais_Proc', 'FIS', 'Fecha_Diagnostico', 'Fecha_Recuperado', 'Fecha_Reporte', 'Tipo_Recuperacion', 'Cod_Depto', 
                    'Cod_Pais', 'Etnia', 'Grupo_Etnico', 'Fecha_Muerte']
# se hace una lista con el conteo de casos por ciudad
ciudades = list(df_casos.Ciudad.value_counts().head().index)
#%%

# Se crea la tabla ciudad sera generica dependiendo de la ciudad seleccionada
# Se define Total para la suma acumulada de casos por ciudad
# Se identifican los casos recuperados y fallecidos para restalos al total de casos, dando como resultado los casos activos
# Se define GR como cambio porcentual de los casos totales por ciudad

def tabla_ciudad(ciudad):
    df = df_casos.loc[(df_casos.Ciudad == ciudad), :]
    dff = pd.crosstab(df['Fecha'], df['Ciudad'])
    dff['Total'] = dff[ciudad].cumsum()
    df_rec = df.loc[(df.Ciudad == ciudad) & ((df.Estado == 'Recuperado') | (df.Estado == 'Fallecido')), :]
    df_recf = pd.crosstab(df_rec['Fecha'], df_rec['Estado'])
    df_recf['Salidas'] = df_recf.Recuperado + df_recf.Fallecido
    dff = pd.merge(dff, df_recf, left_index=True, right_index=True)
    dff['Activos'] = dff['Total'] - dff['Salidas']
    del df_rec, df_recf
    dff['GR'] = dff['Total'].pct_change()
    return dff
#Modelo SIR
#Definicion de variables a Usar para el modelo SIR
# Se define N como la población por ciudad.
# Se dan los parametros iniciales para la población infectada y recuperada
# Se define S como la población suceptible de infectarse
# Se define la tasa de infeccion como "Beta" dada por una media movil de 7 dias
# Se define la tasa de recuperacion como "Gamma" como media movil de 7 dias
# Se define el horizonte de tiempo "t" como el tamaño de la tabla ciudad (dias) + el numero de dias de la predicción (15 dias)
def SIR(ciudad, df_pob, n_pred):
    df = tabla_ciudad(c)
    N = int(df_pob.loc[(df_pob['Grupos de edad'] == 'Total') & (df_pob['Región'] == c), ['Ambos Sexos']].sum() * 0.6)
    I0, R0 = df['Total'][-1], df['Recuperado'].sum()
    S0 = N - I0 - R0
    df['Gamma'] = df['Recuperado'] / df['Total'].shift(1)
    beta, gamma = df['GR'].rolling(7).mean()[-1], df['Gamma'].rolling(7).mean()[-1]
#    beta, gamma = 0.05, 0.01
    t = np.linspace(df.shape[0] + 1, df.shape[0] + n_pred, n_pred)
    
    #Se Calcula primera derivada de los casos suceptibles, infectados y recuperados. Para hallar las variaciones.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    #Se definen los datos iniciales para Infectados, recuperados y suceptibles
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R

#Modelo Gompertz
#
pred = 15
for c in ciudades:
  
    df = tabla_ciudad(c)
    
    def gompertz(x, a, b, c):
        return c * np.exp(-b * np.exp(-x / a))
    def f_gompertz(x, a, b, c):
        return a * (-b) * (-c) * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x)
    
    x = np.arange(0, df.shape[0])
    pred_x = np.arange(df.shape[0] + 1, df.shape[0] + pred)
    y = np.array(df['Activos'])
    f_y = np.array(df[c])
    
    param, pcov = curve_fit(gompertz, x, y, maxfev=10000)
    f_param, f_pcov = curve_fit(f_gompertz, x, f_y, maxfev=10000)
    """Gompertz Casos Totales"""
    plt.plot(x, y, 'b-', label='data')
    plt.plot(x, gompertz(x, *param), 'r--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(param))
    plt.plot(pred_x, gompertz(pred_x, *param), 'g-.', label='Gompertz {} días'.format(pred))
    """SIR Casos Totales"""
    S, I, R = SIR(c, df_pob, pred-1)
    
    # se parametriza los graficos de los Modelos Gompertz y SIR
    plt.plot(pred_x, I, 'm-.', label='SIR {} días'.format(pred))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(c + ' Casos Acumulados')
    plt.show()
    """Gompertz Casos Nuevos"""
    plt.plot(x, f_y, 'b-', label='data')
    plt.plot(x, f_gompertz(x, *f_param), 'r--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(f_param))
    plt.plot(pred_x, f_gompertz(pred_x, *f_param), 'g-.', label='Gompertz {} días'.format(pred))
    """SIR Casos Nuevos"""
    SIR_n = [0 if i < 0 else i for i in np.diff(I)]
    plt.plot(pred_x[1:], SIR_n, 'm-.', label='SIR {} días'.format(pred))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(c + ' Casos Nuevos')
    plt.show()