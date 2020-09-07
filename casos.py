# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:34:06 2020

@author: LENOVO
"""

import pandas as pd
from sodapy import Socrata

def data(lim):
    
    client = Socrata("www.datos.gov.co", None)
      
    casos_api = client.get("gt2j-8ykr", limit=lim)
    df_casos = pd.DataFrame.from_records(casos_api)

    del casos_api
      
    return df_casos
    
df_casos = data(10000000)

df_casos.columns = ['ID', 'Fecha', 'Cod_Municipio', 'Ciudad', 'Depto', 'Estado', 'Edad', 'Sexo', 'Tipo', 'Gravedad',
                   'Pais_Proc', 'FIS', 'Fecha_Diagnostico', 'Fecha_Recuperado', 'Fecha_Reporte', 'Tipo_Recuperacion', 'Cod_Depto', 
                    'Cod_Pais', 'Etnia', 'Grupo_Etnico', 'Fecha_Muerte']

a = df_casos.groupby(['Ciudad','Edad', 'Estado']).size().reset_index()
ciud = ['Bogotá D.C.', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena de Indias']
c = pd.DataFrame(ciud,columns=['ciudades'])
z = pd.merge(a,c,left_on='Ciudad',right_on='ciudades')
z= z[(z['Estado'] != 'Fallecido') & (z['Estado'] != 'N/A') & (z['Estado'] != 'Recuperado')]
z.columns=['Ciudad','Grupo de edad','Estado','Total','ciudades']
z['Grupo de edad'] = pd.cut(z['Grupo de edad'].astype(int),bins=[0,9,19,39,59,79,100], right=False)
z['Estado'] = z['Estado'].replace(['Hospital UCI'], 'Hospital')
desc_casos_est = z.groupby(['Ciudad','Grupo de edad','Estado'])['Total'].sum().reset_index()
desc_casos_est['Grupo de edad'] = desc_casos_est['Grupo de edad'].replace(['[0, 9)'], '0-9')
desc_casos_est.to_csv('Descrip_casos.csv', index=False, encoding= 'ISO-8859-1')

h = df_casos.groupby(['Ciudad','Edad', 'Sexo']).size().reset_index()
ciud = ['Bogotá D.C.', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena de Indias']
c = pd.DataFrame(ciud,columns=['ciudades'])
j = pd.merge(h,c,left_on='Ciudad',right_on='ciudades')
j= j[(j['Sexo'] != 'f') & (j['Sexo'] != 'm') ]
j.columns=['Ciudad','Grupo de edad','Sexo','Total','ciudades']
j['Grupo de edad'] = pd.cut(j['Grupo de edad'].astype(int),bins=[0,9,19,39,59,79,100], right=False)
desc_casos_sexo = j.groupby(['Ciudad','Grupo de edad','Sexo'])['Total'].sum().reset_index()
desc_casos_sexo['Grupo de edad'] = desc_casos_sexo['Grupo de edad'].replace(['[0, 9)'], '0-9')
desc_casos_sexo.to_csv('desc_casos_sexo.csv', index=False, encoding= 'ISO-8859-1')