# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:10:09 2020

@author: daya&santi
"""
import pandas as pd

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

df_pob = poblacion('poblacion_municipios.xlsx')
df_pob['Región'] = df_pob['Región'].replace('Bogotá, D.C.', 'Bogotá D.C.')
df_pob['Región'] = df_pob['Región'].replace('Cartagena', 'Cartagena de Indias')
df_pob['Grupos de edad'] = df_pob['Grupos de edad'].replace(['00-04', '05-09'], '0-9')
df_pob['Grupos de edad'] = df_pob['Grupos de edad'].replace(['10-14', '15-19'], '10-19')
df_pob['Grupos de edad'] = df_pob['Grupos de edad'].replace(['20-24', '25-29', '30-34', '35-39'], '20-39')
df_pob['Grupos de edad'] = df_pob['Grupos de edad'].replace(['40-44', '45-49', '50-54', '55-59'], '40-59')
df_pob['Grupos de edad'] = df_pob['Grupos de edad'].replace(['60-64', '65-69', '70-74', '75-79'], '60-79')
df_pob['Grupos de edad'] = df_pob['Grupos de edad'].replace(['80-84', '85-89', '90-94', '95-99', '100 AÑOS Y MÁS' ], '80 y más')
df_pob

x = df_pob.groupby(['Grupos de edad','Región', 'Ambos Sexos']).size().reset_index()
ciud = ['Bogotá D.C.', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena de Indias']
c = pd.DataFrame(ciud,columns=['ciudades'])
z = pd.merge(x,c,left_on='Región',right_on='ciudades')
z = z[z['Grupos de edad']!= 'Total']
descrip_pob = z.groupby(['Grupos de edad','ciudades'])['Ambos Sexos'].sum().reset_index()
descrip_pob.columns=['Grupos de edad','Region','Total']
descrip_pob.to_csv('Descrip_pob.csv', index=False, encoding= 'ISO-8859-1')


descrip_pob
