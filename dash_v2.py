# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 08:37:50 2020

@author: tomvc
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
#import dash_table_FormatTemplate as FormatTemplate
#import datetime as dt
import plotly.graph_objects as go

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
suppress_callback_exceptions = True

def data():
    df = pd.read_csv(r'C:\Users\tomvc\Desktop\Maestria\Analitica_Predictiva\covid-19-team-ap\BD.csv', 
                     delimiter=',', encoding='ISO-8859-1')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df

def mae():
    return pd.read_csv(r'C:\Users\tomvc\Desktop\Maestria\Analitica_Predictiva\covid-19-team-ap\mae.csv', 
                     delimiter=',', encoding='ISO-8859-1')
def r2():
    return pd.read_csv(r'C:\Users\tomvc\Desktop\Maestria\Analitica_Predictiva\covid-19-team-ap\r2.csv', 
                     delimiter=',', encoding='ISO-8859-1')

def index():
    df = data()
    df_mae = mae()
    return html.Div(
                [
                    dbc.Row([
                            dbc.Col(html.Div(html.H1("Dashboard COVID-19 Ciudades Colombia - Team AP")), width=10),
                            ], justify = 'center'),
                    dbc.Row(dbc.Col(html.Div(html.Hr()))),
                    dbc.Row([
                            dbc.Col(html.Div(dcc.Dropdown(id='ciudad',
                                                          value='Medellín',
                                                          options=[{'label' : i, 'value' : i} for i in df['Ciudad'].unique()],
                                                          searchable=False,
                                                          clearable=False)), width=6),
                            dbc.Col(html.Div(dcc.DatePickerSingle(id='fecha',
                                                                  display_format='D-M-Y',
                                                                  persistence=False)), width = 4)
                            ], justify = 'center'),
                    dbc.Row(dbc.Col(html.Div(html.H1(" ")))),
                    dbc.Row([
                            dbc.Col(dbc.Card([
                                    dbc.CardBody([
                                            html.H6('Medidas de Ajuste y Pronóstico', className='card-title'),
                                            dcc.RadioItems(id='variable',
                                                           options=[{'label' : i, 'value' : i} for i in df_mae.Variable.unique()],
                                                           value='Nuevos',
                                                           labelStyle={'display' : 'inline-block'}),
                                            dash_table.DataTable(id='medidas',
                                                                 style_cell={'textAlign' : 'center', 'fontSize' : 14, 'font-family' : 'arial'})                                            
                                            ])
                                    ]), width=4),
                            dbc.Col(dbc.Card([
                                    dbc.CardBody([
                                            dcc.RadioItems(id='escenario',
                                                           options=[{'label' : 'Base', 'value' : 'Base'},
                                                                    {'label' : 'Sin Cuarentena', 'value' : 'Sin Cuarentena'}],
                                                                    value='Base',
                                                                    labelStyle={'display' : 'inline-block'}),
                                            dcc.Graph(id='grafico'),
                                            ]),
                                    ]), width=6)
                            ], justify = 'center')
                ]
            )

app.layout = index

#Fechas del DatePicker
@app.callback([Output('fecha', 'date'), Output('fecha', 'min_date_allowed'), Output('fecha', 'max_date_allowed')],
               [Input('ciudad', 'value')])

def fechas(ciudad):
    df = data()
    _ = df.loc[(df.Ciudad == ciudad) & (df.Tipo == 'Real'), ['Fecha']]
    f = _.Fecha.max()
    return f, df.Fecha.min(), df.Fecha.max()
    
#Tablas y graficas --> Ciudad y Variable
@app.callback([Output('medidas', 'data'), Output('medidas', 'columns')],
               [Input('ciudad', 'value'), Input('variable', 'value')])

def tabla_medidas(ciudad, variable):
    df_mae = mae()
    df_mae['Medida'] = 'MAE'
    df_r2 = r2()
    df_r2['Medida'] = 'R^2'
    df = pd.concat([df_r2, df_mae])
    df = df.loc[(df.Ciudad == ciudad) & (df.Variable == variable), :]
    df = df[['Medida', 'Valor']]
    df.Valor = df.Valor.apply(lambda x: round(x, 2))
    return df.to_dict('rows'), [{'name' : i, 'id' : i} for i in df.columns]

#Grafico ---> Variable y Ciudad

@app.callback(Output('grafico', 'figure'),
               [Input('ciudad', 'value'), Input('variable', 'value'), Input('escenario', 'value')])

def grafico_ppal(ciudad, variable, escenario):

    df = data()
    df = df.loc[(df.Ciudad == ciudad) & (df.Variable == variable), :]
    
    if escenario == 'Base':
        x = df[df['Tipo'] == 'Real']['Fecha']
        y = df[df['Tipo'] == 'Real']['Valor']
        y_a = df[df['Tipo'] == 'Ajuste']['Valor']
        x_p = df[df['Tipo'] == 'Pronostico']['Fecha']
        y_p = df[df['Tipo'] == 'Pronostico']['Valor']
        
        return go.Figure(data=[
                go.Scatter(x=x,
                           y=y,
                           name='Datos Reales',
                           ),
                go.Scatter(x=x,
                           y=y_a,
                           name='Ajuste'),
                go.Scatter(x=x_p,
                           y=y_p,
                           name='Pronóstico')
                ])
        
        

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')