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
import scipy.stats as st
from pmdarima import auto_arima 
import pmdarima as pm
from sodapy import Socrata
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly.offline import plot
import plotly.graph_objects as go
    

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
    dff = pd.crosstab(df['Fecha_Diagnostico'], df['Ciudad'])
    dff['Total'] = dff[ciudad].cumsum()
    df_rec = df.loc[(df.Ciudad == ciudad) & ((df.Estado == 'Recuperado') | (df.Estado == 'Fallecido')), :]
    df_recf = pd.crosstab(df_rec['Fecha_Diagnostico'], df_rec['Estado'])
    df_recf['Salidas'] = df_recf.Recuperado + df_recf.Fallecido
    dff = pd.merge(dff, df_recf, left_index=True, right_index=True)
    dff['Salidas_total'] = dff['Salidas'].cumsum()
    dff['Activos'] = dff['Total'] - dff['Salidas_total']
    del df_rec, df_recf
    dff['GR'] = dff[ciudad] / dff['Activos'].shift(1)
    # dff['GR'] = dff['Activos'].pct_change()
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
    I0, R0 = df['Activos'][-1], df['Recuperado'].sum()
    S0 = N - I0 - R0
    df['Gamma'] = df['Recuperado'] / df['Activos'].shift(1)
    # beta, gamma = df['GR'].rolling(7).mean()[-1], df['Gamma'].rolling(7).mean()[-1]
    beta, gamma = 0.05, 0.02
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

def arima_fit(residuales, x):
    stepwise_fit = auto_arima(residuales, start_p = 1, start_q = 1, 
                              max_p = 5, max_q = 5, 
                              seasonal = False, 
                              d = None, D = None, trace = False, 
                              error_action ='ignore',   # we don't want to know if an order does not work 
                              suppress_warnings = True,  # we don't want convergence warnings 
                              stepwise = True,
                              information_criterion = 'aic')           # set to stepwise 
    
    
    # se ajusta el modelo
    arma = SARIMAX(residuales,  
                order = stepwise_fit.order)
    
    
    resultado = arma.fit() 
    resultado.summary() 
    
    return resultado


#Modelo Gompertz
#

pred = 30
for c in ciudades:
  
    df = tabla_ciudad(c)
    
    def gompertz(x, a, b, c):
        return c * np.exp(-b * np.exp(-x / a))
    def f_gompertz(x, a, b, c):
        return a * (-b) * (-c) * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x)
    
    # variables de tiempo
    
    x = np.arange(0, df.shape[0])
    pred_x = np.arange(df.shape[0], df.shape[0] + pred)
    y = np.array(df['Total'])
    f_y = np.array(df[c])
    
    # Ajuste modelos
    param, pcov = curve_fit(gompertz, x, y, maxfev=10000)
    f_param, f_pcov = curve_fit(f_gompertz, x, f_y, maxfev=10000)
    """Gompertz Casos Totales"""
    # plt.plot(x, y, 'b-', label='data')
    # plt.plot(x, gompertz(x, *param), 'r--',
    #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(param))
    # plt.plot(pred_x, gompertz(pred_x, *param), 'g-.', label='Gompertz {} días'.format(pred))

    # Ajuste ARMA sobre los residuales
    res = f_y - f_gompertz(x, *f_param)
    arima = arima_fit(res,x)
    aj_arima = arima.predict(min(x), max(x), 
                          typ = 'levels')
    aj_final = f_gompertz(x, *f_param) + aj_arima

    #pronóstico gumpertz + arima
    pron_arima = arima.predict(min(pred_x), max(pred_x), 
                             typ = 'levels')
    
    pronostico = f_gompertz(pred_x, *f_param) + pron_arima

    # nuevos residuales e intervalos de predicción
    n_res = f_y - aj_final
    s = np.std(n_res)
    # límite superior e inferior de los intervalos
    l_s = pronostico + st.norm.ppf(.95) * s
    l_i = pronostico - st.norm.ppf(.95) * s   
    
    
    
    """SIR Casos Totales"""
    S, I, R = SIR(c, df_pob, pred)    

    """SIR Casos Nuevos"""
    SIR_n = [0 if i < 0 else i for i in np.diff(I)]
    
    SIR_n = [0] + SIR_n

    p = np.linspace(0, 1, 15)
    # p = np.zeros(15)
    # p = np.repeat(1,15)
    # Pronóstico a partir del día 15
    pron_final = pronostico[15:]*(1-p) + SIR_n[15:]*p
    
    
    pron_final = np.append(pronostico[:15], pron_final)



    fig = go.Figure()
    # serie original
    fig.add_trace(go.Scatter(x=x, y=f_y,
        name='Casos Reales Diarios', 
        fill=None,
        mode='lines',
        line=dict(width=3)
        ))
    # ajuste
    fig.add_trace(go.Scatter(x=x, y=aj_final,
        name='Modelo Ajustado',
        line = dict(color='red', width=1)
        ))
    
    # intervalo superior
    fig.add_trace(go.Scatter(x=pred_x, y=l_s,
        showlegend=False,
        fill=None,
        mode='lines',
        line=dict(width=0.5, color='rgb(127, 166, 238)'),
        ))
    # intervalo inferior
    fig.add_trace(go.Scatter(
        name='Intervalo de Predicción',
        x=pred_x,
        y=l_i,
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines',
        line=dict(width=0.5, color='rgb(127, 166, 238)')))
    
    # Pronóstico
    fig.add_trace(go.Scatter(x=pred_x, y=pron_final,
        name='Pronósticos Puntuales',
        line = dict(color='royalblue', width=3, dash='dash')
        ))
    
    fig.update_layout(shapes=[
        dict(
          type= 'line',
          yref= 'paper', y0= 0, y1= 1,
          xref= 'x', x0= max(x)+1, x1= max(x)+1,
          fillcolor="rgb(102,102,102)",
          opacity=0.5,
          layer="below",
          line_width=1,
          line=dict(dash="dot")
        )
    ])
    plot(fig)



    casos_n = np.append(aj_final, pron_final)
    casos_t = casos_n.cumsum()
    
    plt.plot(x, y, 'b-', label='data')
    plt.plot(casos_t)


    df['Activos'].plot()
    

    muertos_t = casos_n * 0.01
    df_muertos = df['Fallecido'].cumsum()
    
    
    plt.plot(x, df_muertos, 'b-', label='data')
    plt.plot(muertos_t)











    # # se parametriza los graficos de los Modelos Gompertz y SIR
    # plt.plot(pred_x, I, 'm-.', label='SIR {} días'.format(pred))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.title(c + ' Casos Acumulados')
    # plt.show()
    # """Gompertz Casos Nuevos"""
    # plt.plot(x, f_y, 'b-', label='data')
    # plt.plot(x, f_gompertz(x, *f_param), 'r--',
    #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(f_param))
    # plt.plot(pred_x, f_gompertz(pred_x, *f_param), 'g-.', label='Gompertz {} días'.format(pred))



    # plt.plot(pred_x[1:], SIR_n, 'm-.', label='SIR {} días'.format(pred))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.title(c + ' Casos Nuevos')
    # plt.show()