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
    df = pd.read_csv(r'BD.csv', 
                     delimiter=',', encoding='ISO-8859-1')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df

def mae():
    return pd.read_csv(r'mae.csv', 
                     delimiter=',', encoding='ISO-8859-1')
def r2():
    return pd.read_csv(r'r2.csv', 
                     delimiter=',', encoding='ISO-8859-1')

def index():
    df = data()
    df_mae = mae()
    return html.Div(
                [
                    dbc.Row([
                            dbc.Col(html.Div(html.H1("Predicción COVID-19 Ciudades Colombia - Team AP")), width=10),
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
                                                                 style_cell={'textAlign' : 'center', 'fontSize' : 14, 'font-family' : 'arial'}),
                                            html.H1(' '),
                                            html.H6(id='uci-title'),
                                            dcc.Graph(id='uci-graf')
                                            ])
                                    ]), width=4),
                            dbc.Col(dbc.Card([
                                    dbc.CardBody([
                                            html.H6('Pronóstico por Escenarios de Restricciones', className='card-title'),
                                            dcc.RadioItems(id='escenario',
                                                           options=[{'label' : 'Estrictas', 'value' : 'Base'},
                                                                    {'label' : 'Parciales', 'value' : 'USA'},
                                                                    {'label' : 'Sin Restricciones', 'value' : 'Mexico'}],
                                                                    value='Base',
                                                                    labelStyle={'display' : 'inline-block'}),
                                            dcc.Graph(id='grafico'),
                                            html.P(id='exp'),
                                            ]),
                                    ]), width=6)
                            ], justify = 'center'),
                    dbc.Row(dbc.Col(html.Div(html.H1(" ")))),
                    dbc.Row([
                            dbc.Col(dbc.Card([
                                    dbc.CardBody([
                                            html.H6('Descripción Casos Covid por Rangos de Edad'),
                                            dcc.RadioItems(id='covid',
                                                           options=[{'label' : 'Género', 'value' : 'genero'},
                                                                    {'label' : 'Lugar de Recuperación', 'value' : 'tratamiento'}],
                                                                    value='genero',
                                                                    labelStyle={'display' : 'inline-block'}),
                                            dcc.Graph(id='covid-graf')
                                            ])
                                    ]), width=10)
                            ], justify='center'),
                    dbc.Row(dbc.Col(html.Div(html.Hr()))),
                    dbc.Row([
                            dbc.Col(html.Div(html.Footer('Equipo: Dayana Jiménez Vanegas, Santiago Agudelo Martínez, Julian Palacio Roldan y Tomás Vergara Cardona')), width=10)
                            ], justify = 'end')
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
    return f, f, f
    
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
    df.Valor = df.Valor.apply(lambda x: round(x, 3))
    return df.to_dict('rows'), [{'name' : i, 'id' : i} for i in df.columns]

#Informacion UCI por Ciudad
    
@app.callback([Output('uci-title', 'children'), Output('uci-graf', 'figure')],
              [Input('ciudad', 'value')])

def uci(ciudad):
    df = pd.read_csv(r'Ocupacion UCI.csv', 
                     delimiter=',', encoding='ISO-8859-1')
    df = df.loc[df.Ciudad == ciudad, :]
    oc = str(df.iloc[0]['Ocupacion'])
    label=['Pacientes Covid', 'Sospechosos Covid', 'No Covid']
    return 'Ocupación UCI {}: {}'.format(ciudad, oc), go.Figure(data=[
                                                        go.Pie(labels=label,
                                                               values=[df.iloc[0][j] for j in label],
                                                               name='Uso UCI',
                                                               pull=[0.1, 0, 0])
                                                        ])

#Grafico ---> Variable y Ciudad

@app.callback(Output('grafico', 'figure'),
               [Input('ciudad', 'value'), Input('variable', 'value'), Input('escenario', 'value')])

def grafico_ppal(ciudad, variable, escenario):
    
    df = data()
    df = df.loc[(df.Ciudad == ciudad) & (df.Variable == variable), :]
    df.sort_values('Fecha', inplace=True)
    
    x = df[df['Tipo'] == 'Real']['Fecha']
    y = df[df['Tipo'] == 'Real']['Valor']
    y_a = df[df['Tipo'] == 'Ajuste']['Valor']
    x_p = df[(df['Tipo'] == 'Pronostico') & (df.Escenario == escenario)]['Fecha']
    y_p = df[(df['Tipo'] == 'Pronostico') & (df.Escenario == escenario)]['Valor']
    li = df[(df['Tipo'] == 'LI') & (df.Escenario == escenario)]['Valor']
    ls = df[(df['Tipo'] == 'LS') & (df.Escenario == escenario)]['Valor']
    
    fig = go.Figure(data=[
            go.Scatter(x=x,
                       y=y,
                       name='Datos Reales',
                       ),
            go.Scatter(x=x,
                       y=y_a,
                       name='Ajuste'),
            go.Scatter(x=x_p,
                       y=y_p,
                       name='Pronóstico'),
            go.Scatter(x=x_p,
                       y=ls,
                       showlegend=False,
                       fill=None,
                       mode='lines',
                       line=dict(width=0.5, color='rgb(127, 166, 238)')
                       ),
            go.Scatter(x=x_p,
                       y=li,
                       fill='tonexty',
                       mode='lines',
                       name='Intervalo de Predicción',
                       line=dict(width=0.5, color='rgb(127, 166, 238)')),
            ])
    fig.update_layout(
        title={
            'text': ciudad + ' - ' + variable,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    
    return fig

#Explicacion escenario
@app.callback(Output('exp', 'children'), [Input('escenario', 'value')])

def explicacion(esc):
    if esc == 'Base':
        return """Se estima el comportamiento de la pandemia por medio del modelo 
    Gompertz y ARIMA bajo la implementación de medidas estrictas de seguridad frente a la pandemia."""
    elif esc == 'Mexico':
        return """Se utilizaron las tasas de infección y recuperación de México para predecir el comportamiento 
    de Colombia bajo condiciones sin restricciones aéreas y pocas medidas de seguridad frente a la pandemia."""
    elif esc == 'USA':
        return """Se utilizaron las tasas de infección y recuperación de los Estados Unidos 
            para predecir el comportamiento de Colombia bajo algunas medidas de seguridad frente a la pandemia."""

#Descripciones COVID
@app.callback(Output('covid-graf', 'figure'),
              [Input('covid', 'value'), Input('ciudad', 'value')])

def covid_graf(tipo, ciudad):
#    ciudad = 'Cali'
    if tipo == 'genero':
        df = pd.read_csv(r'desc_casos_sexo.csv', 
                     delimiter=',', encoding='ISO-8859-1')
        df = df.loc[(df.Ciudad == ciudad), :]
        x = df['Grupo de edad'].unique()
        y_m = df[df['Sexo'] == 'F']['Total']
        y_h = df[df['Sexo'] == 'M']['Total']
        fig = go.Figure(data=[
            go.Bar(name='F', x=x, y=y_m),
            go.Bar(name='M', x=x, y=y_h),
            ])
        fig.update_layout(barmode='stack')
        fig.update_layout(
            title={
                'text': ciudad,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        return fig
    else:
        df = pd.read_csv(r'Descrip_casos.csv', 
                     delimiter=',', encoding='ISO-8859-1')
        df = df.loc[(df.Ciudad == ciudad), :]
        x = df['Grupo de edad'].unique()
        y_m = df[df['Estado'] == 'Casa']['Total']
        y_h = df[df['Estado'] == 'Hospital']['Total']
        fig = go.Figure(data=[
            go.Bar(name='Casa', x=x, y=y_m),
            go.Bar(name='Hospital', x=x, y=y_h),
            ])
        fig.update_layout(barmode='stack')
        fig.update_layout(
            title={
                'text': ciudad,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        return fig


if __name__ == "__main__":
    
    import os
            
    if os.environ.get("deploy_en_heroku") == 'TRUE':
        port = int(os.environ.get("PORT", 5000))
        app.run_server(host='0.0.0.0',debug=False, port=port)
        
    else:
        app.run_server(host='0.0.0.0',debug=True, port=8050)