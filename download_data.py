# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:40:53 2020

@author: tomvc
"""

import pandas as pd
from sodapy import Socrata

def data(lim):
    
    client = Socrata("www.datos.gov.co", None)
    
    pruebas_api = client.get("8835-5baf", limit=lim)
    df_pruebas = pd.DataFrame.from_records(pruebas_api)
    
    del pruebas_api
    
    casos_api = client.get("gt2j-8ykr", limit=lim)
    df_casos = pd.DataFrame.from_records(casos_api)

    del casos_api
      
    return df_pruebas, df_casos



def poblacion(ruta):
    # Lectura de datos
    df = pd.read_excel(ruta, sheet_name='PPO_GQEdad_DPTO')
    
    # se extraen los códigos de cada departamento
    codigos = df[df['Codigo'].notna()][['Codigo','Grupos de edad']]
    codigos.columns = ['Codigo','Región']
    
    # se llenan los NA con el código inmediatamente anterior
    df['Codigo'] = df['Codigo'].fillna(method='ffill')
    
    # se eliminan las filas de los títulos de los departamentos
    # y las filas del final que traen NA
    df = df.dropna()
    
    # se hace un merge para tener los nombres de los departamentos
    df = pd.merge(df,codigos,on='Codigo',how='left')
    
    return df

df = poblacion('poblacion.xlsx')