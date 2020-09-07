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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import math
from sklearn.metrics import mean_squared_error


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
    df['FIS'] = np.where(df['FIS'] == 'Asintomático',df['Fecha'] ,df['FIS'])
    df['FIS'] = [dt.datetime.combine(dt.datetime.strptime(i,'%Y-%m-%dT%H:%M:%S.%f').date(), dt.datetime.min.time()) for i in df['FIS']]


    # df['FIS'] = pd.to_datetime(df['FIS'], format='%Y-%m-%d')
    
    # df['FIS'] = pd.to_datetime(df['FIS'], format= '%Y-%m-%dT%H:%M:%S.%f')
    # dt.datetime(strptime(i, format='%Y-%m-%dT%H:%M:%S.%f'))
    
    dff = pd.crosstab(df['FIS'], df['Ciudad'])
    dff['Total'] = dff[ciudad].cumsum()
    df_rec = df.loc[(df.Ciudad == ciudad) & ((df.Estado == 'Recuperado') | (df.Estado == 'Fallecido')), :]
    df_recf = pd.crosstab(df_rec['Fecha_Recuperado'], df_rec['Estado'])
    df_m = pd.crosstab(df_rec['Fecha_Muerte'], df_rec['Estado'])
    df_s = pd.merge(df_recf, df_m['Fallecido'], left_index=True, right_index=True, how='left')
    df_s.fillna(0, inplace=True)
    df_s['Salidas'] = df_s.Recuperado + df_s.Fallecido
    dff = pd.merge(dff, df_s, left_index=True, right_index=True, how = 'left')
    dff.fillna(0, inplace=True)
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
def SIR(ciudad, df_pob, n_pred, beta, gamma):

    df = tabla_ciudad(c)
    N = int(df_pob.loc[(df_pob['Grupos de edad'] == 'Total') & (df_pob['Región'] == c), ['Ambos Sexos']].sum() * 0.6)
    I0, R0 = df['Total'][-1], df['Recuperado'].sum()
    S0 = N - I0 - R0
    df['Gamma'] = df['Recuperado'] / df['Activos'].shift(1)
    # beta, gamma = df['GR'].rolling(7).mean()[-1], df['Gamma'].rolling(7).mean()[-1]
    # beta, gamma = 0.05, 0.02
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

def mod_arima(datos, x, pred_x):
    
    stepwise_fit = auto_arima(datos, seasonal = False,trace=False,start_p=5)   
    
    # se ajusta el modelo
    arma = SARIMAX(datos,  
                order = stepwise_fit.order)
    
    
    resultado = arma.fit() 
    resultado.summary() 
    
    ajuste = resultado.predict(min(x), max(x), 
                          typ = 'levels')
    
    pronostico = resultado.predict(min(pred_x), max(pred_x), 
                          typ = 'levels')
    
    
    return ajuste, pronostico

def mod_gompertz(y, x, pred_x, tipo='nuevos'):

    y = list(y)

    #Modelo Gompertz casos totales
    def gompertz(x, a, b, c):
        return c * np.exp(-b * np.exp(-x / a))
    # Modelo Gompertz casos nuevos (Derivada)
    def f_gompertz(x, a, b, c):
        return (a * (-b) * (-c) * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x))
    def gauss(x, s,u,c):
        return (1/(s*math.sqrt(2*math.pi)))*np.exp((-1*(x-u)**2)/(2*s**2))*c
    


    # Ajuste y pronóstico
    if tipo == 'nuevos':
        f_param, pcov = curve_fit(f_gompertz, x, y, maxfev=10000000,p0=([1000,1000,10]))
        ajuste = f_gompertz(x, *f_param)
        pronostico = f_gompertz(pred_x, *f_param)
        f_param2, pcov2 = curve_fit(gauss, x, y, maxfev=10000000, bounds=([10,100,1000],[10000,10000,10000000]))
        ajuste2 = gauss(x, *f_param2)
        pronostico2 = gauss(pred_x, *f_param2)
        
        
        mse1 = mean_squared_error(y,ajuste)
        mse2 = mean_squared_error(y,ajuste2)
        
        if mse2 <= mse1:
            ajuste = ajuste2
            pronostico = pronostico2
        
        # plt.plot(y)    
        # plt.plot(ajuste, 'r-')
        # plt.plot(pred_x,pronostico, 'r-')
        # # plt.plot(ajuste2, 'g-')
        # # plt.plot(pred_x,pronostico2, 'g-')
        # plt.title(f'Ciudad: {c}, variable: {var}')
        # plt.show()        

        
    elif tipo == 'totales':
        param, pcov = curve_fit(gompertz, x, y, maxfev=10000000)
        ajuste = gompertz(x, *param)
        pronostico = gompertz(pred_x, *param)
        

        ajuste2 = np.repeat(0,len(x))
        pronostico2 = np.repeat(0,len(pred_x))
    else:
        print('Elija un tipo válido')
        
        
    
    return ajuste, pronostico

# número de pronósticos
pred = 45
# datos para la validación cruzada
val_cruz = 5

# Variables para bodega de datos
fmae = []
fr2 = []
n = []
f = []
u = []
v = []
e = []
g = []
esc = []

for c in ciudades:

    df = tabla_ciudad(c)
    df.index = pd.to_datetime(df.index)
    
    # c = 'Bogotá D.C.'

    # =============================================================================
    # variables de tiempo
    # =============================================================================
    
    N = df.shape[0]
    ff = df.index.max()
    fi = df.index.min()
    fechas = list(df.index)
    
    # Serie completa
    x = np.arange(0, N)
    pred_x = np.arange(N, N + pred)    
    
    #Validación cruzada
    x_val = np.arange(0, N - val_cruz)
    pred_x_val = np.arange(N - val_cruz, N)
    

    # =============================================================================
    # Casos Totales
    # =============================================================================    
      
    y = np.array(df['Total'])
    
    #Construccion CSV
    for i in range(N):
      n.append(x[i])
      f.append(fechas[i])
      u.append(c)
      v.append(df.iloc[i]['Total'])
      e.append('Total')
      g.append('Real')
      esc.append('Base')
    
    # Ajuste modelos
    """Gompertz Casos Totales"""
    
     ### Validación cruzada

    # ajuste totales gompertz
    aj_t_gom, pron_t_gom = mod_gompertz(y[x_val],x_val,pred_x_val,tipo='totales')

    # Ajuste ARMA sobre los residuales
    res = y[x_val] - aj_t_gom
    aj_arima,pron_arima  = mod_arima(res,x_val,pred_x_val)
    
    # Ajuste gompertz + arima
    aj_t_final = aj_t_gom + aj_arima

    #pronóstico gumpertz + arima
    pron_final = pron_t_gom + pron_arima
    
    mae = mean_absolute_error(y[pred_x_val], pron_final)
    fmae.append([c, 'Total', mae,'Base'])
    

    ### Ahora con todos los datos
    
    aj_t_gom, pron_t_gom = mod_gompertz(y,x,pred_x,'totales')

    # Ajuste ARMA sobre los residuales
    res = y - aj_t_gom
    aj_arima,pron_arima  = mod_arima(res,x,pred_x)
    
    # Ajuste gompertz + arima
    aj_t_final = aj_t_gom + aj_arima

    #pronóstico gumpertz + arima
    pron_t_final = pron_t_gom + pron_arima


    # nuevos residuales e intervalos de predicción
    n_res = y - aj_t_final
    s = np.std(n_res)
    # límite superior e inferior de los intervalos
    l_s = pron_final + st.norm.ppf(.95) * s
    l_i = pron_final - st.norm.ppf(.95) * s   
        
    r2 = r2_score(y, aj_t_final)

    fr2.append([c, 'Total', r2,'Base'])
    
    
    # plt.plot(x,y, 'b-')
    # plt.plot(x,aj_t_final, 'r-' )
    # plt.plot(pred_x,pron_t_final, 'g-')
    # plt.show()
    
    #Construccion CSV
    for i in range(len(aj_t_final)):
      n.append(x[i])
      f.append(fechas[i])
      u.append(c)
      v.append(aj_t_final[i])
      e.append('Total')
      g.append('Ajuste')
      esc.append('Base')

    for i in range(len(pron_t_final)):
      n.append(pred_x[i])
      f.append(fechas[-1] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(pron_t_final[i])
      e.append('Total')
      g.append('Pronostico')
      esc.append('Base')
      
    for i in range(len(l_s)):
      n.append(pred_x[i])
      f.append(fechas[-1] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(l_s[i])
      e.append('Total')
      g.append('LS')
      esc.append('Base')
    
    for i in range(len(l_i)):
      n.append(pred_x[i])
      f.append(fechas[-1] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(l_i[i])
      e.append('Total')
      g.append('LI')
      esc.append('Base')
      
      
    ### Pronósticos con escenarios de apertura
      
    """SIR Casos Totales"""
    # EEUU 0.0167,0.0121
    # México 0.0515,0.0364
    Se, Ie, Re = SIR(c, df_pob, pred, 0.0167,0.0121)    
    Sm, Im, Rm = SIR(c, df_pob, pred, 0.0515,0.0364)
    # se crea un vector donde se reparten los pesos de cada modelo a partir
    # de dos semanas de pronóstico
    p = np.linspace(0, 1, 30)

    # Pronóstico a partir del día 15 USA
    pron_final_usa = pron_t_final[15:]*(1-p) + Ie[15:]*p
    pron_final_usa = np.append(pron_t_final[:15], pron_final_usa)
    
     # Pronóstico a partir del día 15 MÉXICO
    pron_final_mex = pron_t_final[15:]*(1-p) + Im[15:]*p
    pron_final_mex = np.append(pron_t_final[:15], pron_final_mex)   
    
    # escenario México
    for i in range(len(pron_final_mex)):
      n.append(pred_x[i])
      f.append(fechas[-1] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(pron_final_mex[i])
      e.append('Total')
      g.append('Pronostico')
      esc.append('Mexico')

    # escenario  USA
    for i in range(len(pron_final_usa)):
      n.append(pred_x[i])
      f.append(fechas[-1] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(pron_final_usa[i])
      e.append('Total')
      g.append('Pronostico')
      esc.append('USA')
      
    # plt.plot(x,y, 'b-')
    # plt.plot(x,aj_t_final, 'r-' )
    # plt.plot(pred_x,pron_final, 'g-')
    # plt.show()     
    
    # =============================================================================
    # Casos nuevos
    # =============================================================================

    # Variable dependiente
    f_y = np.array(df[c][:-7])
    
    #Construccion CSV
    for i in range(len(df[c][:-7])):
      n.append(x[i])
      f.append(fechas[i])
      u.append(c)
      v.append(df[c].iloc[i])
      e.append('Nuevos')
      g.append('Real')
      esc.append('Base')
    
    
    
    """Gompertz Casos Nuevos"""
    
    ### Validación cruzada

    aj_n_gom, pron_n_gom = mod_gompertz(f_y[x_val[:-7]],x_val[:-7],pred_x_val-7,tipo='nuevos')

    # Ajuste ARMA sobre los residuales
    res = f_y[x_val[:-7]] - aj_n_gom
    aj_arima,pron_arima  = mod_arima(res,x_val[:-7],pred_x_val-7)
    
    # Ajuste gompertz + arima
    aj_n_final = aj_n_gom + aj_arima

    #pronóstico gumpertz + arima
    pron_final = pron_n_gom + pron_arima

    # plt.plot(x[:-7],f_y, 'b-')
    # plt.plot(x_val[:-7],aj_n_final, 'r-' )
    # plt.plot(pred_x_val-7,pron_final, 'g-')
    # plt.title(c)
    # plt.show()


    mae = mean_absolute_error(f_y[pred_x_val-7], pron_final)
    fmae.append([c, 'Nuevos', mae,'Base'])
    
    
    ### Ahora con todos los datos
    
    aj_n_gom, pron_n_gom = mod_gompertz(f_y,x[:-7],pred_x-7)

    # plt.plot(f_y)
    # plt.plot(aj_n_gom, 'g-')
    # plt.show()

    # Ajuste ARMA sobre los residuales
    res = f_y - aj_n_gom
        
    # =============================================================================
    # Análisis de residuales    
    # =============================================================================

    # plt.plot(res)
    # plt.show()

    # pm.plot_acf(res)
    # pm.plot_pacf(res)    

    # stepwise_fit = auto_arima(res, seasonal=False)           # set to stepwise 
        
        
    # # # se ajusta el modelo
    # arma = SARIMAX(res,order=stepwise_fit.order)
    
    
    # resultado = arma.fit() 
    # resultado.summary() 
    
    # ajuste = resultado.predict(1, len(res), 
    #                      typ = 'levels')
    
    # pronostico = resultado.predict(min(pred_x-7), max(pred_x-7), typ = 'levels')
        
    # aj_n_final = aj_n_gom + ajuste

    # pron_n_final = pron_n_gom + pronostico

    # plt.plot(f_y)
    # plt.plot(aj_n_gom, 'g-')
    # plt.plot(ajuste, 'r-')
    
    
    
    
    
    aj_arima,pron_arima  = mod_arima(res,x[:-7],pred_x-7)
    
    # Ajuste gompertz + arima
    aj_n_final = aj_n_gom + aj_arima

    #pronóstico gumpertz + arima
    pron_n_final = pron_n_gom + pron_arima

    for i in range(len(aj_n_final)):
      n.append(x[i])
      f.append(fechas[i])
      u.append(c)
      v.append(aj_n_final[i])
      e.append('Nuevos')
      g.append('Ajuste')
      esc.append('Base')

    for i in range(len(pron_n_final)):
      n.append(pred_x[i])
      f.append(fechas[-8] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(pron_n_final[i])
      e.append('Nuevos')
      g.append('Pronostico')
      esc.append('Base')

    for i in range(len(l_s)):
      n.append(pred_x[i])
      f.append(fechas[-8] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(l_s[i])
      e.append('Nuevos')
      g.append('LS')
      esc.append('Base')
    
    for i in range(len(l_i)):
      n.append(pred_x[i])
      f.append(fechas[-8] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(l_i[i])
      e.append('Nuevos')
      g.append('LI')
      esc.append('Base')

    # nuevos residuales
    n_res = f_y - aj_n_final
    s = np.std(n_res)
 
    r2 = r2_score(f_y, aj_n_final)

    fr2.append([c, 'Nuevos', r2,'Base'])

   
    """SIR Casos Nuevos"""
    # México
    SIR_n_m = [0 if i < 0 else i for i in np.diff(Im)]
    # USA
    SIR_n_u = [0 if i < 0 else i for i in np.diff(Ie)]

    # se agrega un cero para que tenga la misma longitud que el pronóstico del
    # modelo gompertz + arima
    SIR_n_m = [0] + SIR_n_m
    SIR_n_u = [0] + SIR_n_u
    
    # se crea un vector donde se reparten los pesos de cada modelo a partir
    # de dos semanas de pronóstico
    p = np.linspace(0, 1, 30)

    # Pronóstico a partir del día 15 México
    pron_final_usa = pron_n_final[15:]*(1-p) + SIR_n_u[15:]*p
    pron_final_usa = np.append(pron_n_final[:15], pron_final_usa)

    # Pronóstico a partir del día 15 USA
    pron_final_mex = pron_n_final[15:]*(1-p) + SIR_n_m[15:]*p
    pron_final_mex = np.append(pron_n_final[:15], pron_final_mex)

  
    # escenario México
    for i in range(len(pron_final_mex)):
      n.append(pred_x[i])
      f.append(fechas[-8] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(pron_final_mex[i])
      e.append('Nuevos')
      g.append('Pronostico')
      esc.append('Mexico')

    # escenario  USA
    for i in range(len(pron_final_usa)):
      n.append(pred_x[i])
      f.append(fechas[-8] + dt.timedelta(days=i+1))
      u.append(c)
      v.append(pron_final_usa[i])
      e.append('Nuevos')
      g.append('Pronostico')
      esc.append('USA')

    
    # Crecimiento de cada curva para las demás variables
    ## México
    # cambio porcentual por el escenario
    crecimiento_m = pd.DataFrame(pron_final_mex).pct_change()
    crecimiento_m[0] = crecimiento_m[0] + 1
    
    # se dejan los primeros 15 días en 1 (en este periodo no se pretende
    # modificar el pronóstico del modelo a corto plazo)    
    crecimiento_m[0][:15] = 1
    # con el cumprod se halla el cambio porcentual de cada pronóstico con respecto
    # al pronóstico número 15
    crecimiento_m = list(crecimiento_m[0])
    
    ## USA
    # cambio porcentual por el escenario
    crecimiento_u = pd.DataFrame(pron_final_usa).pct_change()
    crecimiento_u[0] = crecimiento_u[0] + 1
    
    # se dejan los primeros 15 días en 1 (en este periodo no se pretende
    # modificar el pronóstico del modelo a corto plazo)    
    crecimiento_u[0][:15] = 1
    # con el cumprod se halla el cambio porcentual de cada pronóstico con respecto
    # al pronóstico número 15
    crecimiento_u = list(crecimiento_u[0])   
    


    # =============================================================================
    #  Casos Activos, Recuperados, Fallecidos
    # =============================================================================

    for var in ['Activos','Recuperado','Fallecido']:

        #Construccion CSV
        for i in range(len(df[var])):
          n.append(x[i])
          f.append(fechas[i])
          u.append(c)
          v.append(df[var].iloc[i])
          e.append(var)
          g.append('Real')
          esc.append('Base')
        
        
        ### Validación cruzada

        aj_gom, pron_gom = mod_gompertz(df[var][x_val], x_val, pred_x_val)
    
        # Ajuste ARMA sobre los residuales
        res = df[var][x_val] - aj_gom
        aj_arima,  pron_arima = mod_arima(res,x_val,pred_x_val)
        
        # ajuste gompertz + arima
        aj_final = aj_gom + aj_arima
    
    
        #pronóstico gumpertz + arima
        pron_final = pron_gom + pron_arima
        
        # MAE
        mae = mean_absolute_error(df[var][pred_x_val], pron_final)       

        # plt.plot(x,df[var], 'b-')
        # plt.plot(x_val,aj_final, 'r-' )
        # plt.plot(pred_x_val,pron_final, 'g-')
        # plt.title(f'Ciudad: {c}, variable: {var}')
        # plt.show()
        
        # bodega de datos
        fmae.append([c, var, mae,'Base'])
        
        ### Modelo con todos los datos

        aj_gom, pron_gom = mod_gompertz(df[var], x, pred_x)
    
        # Ajuste ARMA sobre los residuales
        res = df[var] - aj_gom
        aj_arima,  pron_arima = mod_arima(res,x,pred_x)
        
        # ajuste gompertz + arima
        aj_final = aj_gom + aj_arima
    
    
        #pronóstico gumpertz + arima
        pron_final = pron_gom + pron_arima
    
    
        #nuevos residuales e intervalos de predicción
        n_res = df[var] - aj_final
        s = np.std(n_res)
        # límite superior e inferior de los intervalos
        l_s = pron_final + st.norm.ppf(.95) * s
        l_i = pron_final - st.norm.ppf(.95) * s   
        
        r2 = r2_score(df[var], aj_final)
        fr2.append([c, var, r2,'Base']) 


        # Bodega de datos
        for i in range(len(aj_final)):
          n.append(x[i])
          f.append(fechas[i])
          u.append(c)
          v.append(aj_final[i])
          e.append(var)
          g.append('Ajuste')
          esc.append('Base')

        pron_final = list(pron_final)

        for i in range(len(pron_final)):
          n.append(pred_x[i])
          f.append(fechas[-1] + dt.timedelta(days=i+1))
          u.append(c)
          v.append(pron_final[i])
          e.append(var)
          g.append('Pronostico')
          esc.append('Base')

        l_s = list(l_s)
        l_i = list(l_i) 
        for i in range(len(l_s)):
          n.append(pred_x[i])
          f.append(fechas[-1] + dt.timedelta(days=i+1))
          u.append(c)
          v.append(l_s[i])
          e.append(var)
          g.append('LS')
          esc.append('Base')
    
        for i in range(len(l_i)):
          n.append(pred_x[i])
          f.append(fechas[-1] + dt.timedelta(days=i+1))
          u.append(c)
          v.append(l_i[i])
          e.append(var)
          g.append('LI')
          esc.append('Base')


        #Escenarios
        
        # Pronóstico a partir del día 15 México
        pron_final_mex = []
        anterior = pron_final[14]
        for i in range(15,len(crecimiento_m)):
            anterior = anterior * crecimiento_m[i]
            pron_final_mex.append(anterior)
            
        pron_final_mex = np.append(pron_final[:15], pron_final_mex)
            
  
        # Pronóstico a partir del día 15 USA
        pron_final_usa = []
        anterior = pron_final[14]
        for i in range(15,len(crecimiento_u)):
            anterior = anterior * crecimiento_u[i]
            pron_final_usa.append(anterior)
            
        pron_final_usa = np.append(pron_final[:15], pron_final_usa) 
        
        

          
        # escenario México
        for i in range(len(pron_final_mex)):
          n.append(pred_x[i])
          f.append(fechas[-1] + dt.timedelta(days=i+1))
          u.append(c)
          v.append(pron_final_mex[i])
          e.append(var)
          g.append('Pronostico')
          esc.append('Mexico')
    
        # escenario  USA
        for i in range(len(pron_final_usa)):
          n.append(pred_x[i])
          f.append(fechas[-1] + dt.timedelta(days=i+1))
          u.append(c)
          v.append(pron_final_usa[i])
          e.append(var)
          g.append('Pronostico')
          esc.append('USA')

          
        # plt.plot(x,df[var], 'b-')
        # plt.plot(x,aj_final, 'r-' )
        # plt.plot(pred_x,pron_final, 'g-')
        # plt.title(f'Ciudad: {c}, variable: {var}')
        # plt.show()
        



final = pd.DataFrame(list(zip(n, f, u, v, e, g, esc)), columns=['N', 'Fecha', 'Ciudad', 'Valor', 'Variable', 'Tipo','Escenario'])
df_mae = pd.DataFrame(data=fmae, columns=['Ciudad', 'Variable', 'Valor','Escenario'])
df_r2 = pd.DataFrame(data=fr2, columns=['Ciudad', 'Variable', 'Valor','Escenario'])

final.to_csv('BD.csv', index=False, encoding='ISO-8859-1')
df_mae.to_csv('mae.csv', index=False, encoding='ISO-8859-1')
df_r2.to_csv('r2.csv', index=False, encoding='ISO-8859-1')

      