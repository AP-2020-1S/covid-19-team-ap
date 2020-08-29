# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:29:32 2020

@author: tomvc
"""

import pandas as pd
from download_data import data, poblacion
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import odeint

df_pruebas, df_casos = data(10000000)
df_pob = poblacion('poblacion_municipios.xlsx')
df_pob['Región'] = df_pob['Región'].replace('Bogotá, D.C.', 'Bogotá D.C.')
df_pob['Región'] = df_pob['Región'].replace('Cartagena de Indias', 'Cartagena')
df_casos.columns = ['ID', 'Fecha', 'Cod_Municipio', 'Ciudad', 'Depto', 'Estado', 'Edad', 'Sexo', 'Tipo', 'Gravedad',
                   'Pais_Proc', 'FIS', 'Fecha_Diagnostico', 'Fecha_Recuperado', 'Fecha_Reporte', 'Tipo_Recuperacion', 'Cod_Depto', 
                    'Cod_Pais', 'Etnia', 'Grupo_Etnico', 'Fecha_Muerte']

ciudades = list(df_casos.Ciudad.value_counts().head().index)

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

def SIR(ciudad, df_pob, n_pred):
    df = tabla_ciudad(c)
    N = int(df_pob.loc[(df_pob['Grupos de edad'] == 'Total') & (df_pob['Región'] == c), ['Ambos Sexos']].sum() * 0.6)
    I0, R0 = df['Total'][-1], df['Recuperado'].sum()
    S0 = N - I0 - R0
    beta, gamma = df['GR'][-1], df['Recuperado'].mean() / df['Total'][-3]
    t = np.linspace(df.shape[0] + 1, df.shape[0] + n_pred, n_pred)
    
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R

pred = 30
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
    
    plt.plot(x, y, 'b-', label='data')
    plt.plot(x, gompertz(x, *param), 'r--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(param))
    plt.plot(pred_x, gompertz(pred_x, *param), 'g-.', label='Predicción {} días'.format(pred))
    S, I, R = SIR(c, df_pob, pred-1)
    plt.plot(pred_x, I, 'm-.', label='SIR')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(c + ' Casos Acumulados')
    plt.show()
    
    plt.plot(x, f_y, 'b-', label='data')
    plt.plot(x, f_gompertz(x, *f_param), 'r--',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(f_param))
    plt.plot(pred_x, f_gompertz(pred_x, *f_param), 'g-.', label='Predicción {} días'.format(pred))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(c + ' Casos Nuevos')
    plt.show()