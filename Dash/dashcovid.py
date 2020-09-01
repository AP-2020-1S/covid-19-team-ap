# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:36:33 2020

@author: Julian
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from download_data import data, poblacion
from datetime import datetime
import numpy as np
from pmdarima import auto_arima 
import pmdarima as pm
from scipy.optimize import curve_fit
from sodapy import Socrata
import plotly.graph_objects as go
    
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


pestana1= [html.Div(id="tab-content", className="p-4", children =[
                dcc.Dropdown(options = [
                                        {'label':'Bogotá D.C.', 'value':'Bogotá D.C.'},
                                        {'label':'Medellín', 'value':'Medellín'},
                                        {'label':'Cali', 'value':'Cali'},
                                        {'label':'Barranquilla', 'value':'Barranquilla'},
                                        {'label':'Cartagena de Indias', 'value':'Cartagena de Indias'}],
                            clearable=False,
                            value='Bogotá D.C.',
                            id='drop_ciudad'),
                dcc.Graph(id='graf_casos')
                
            ])]


app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.H1("Pronóstico COVID-19 TeamAP"),
        html.Hr(),
        dcc.Loading(dbc.Button(
            "Actualizar modelo",
            color="primary",
            block=True,
            id="boton_actualizar",
            className="mb-3",
        )),
        dbc.Tabs(
            [
                dbc.Tab(label="Cuarentena Estricta", tab_id="cuarentena", children=pestana1),
                dbc.Tab(label="Apertura Económica y social", tab_id="apertura"),
            ],
            id="tabs",
            active_tab="cuarentena")
        
    ]
)



@app.callback(
    Output("graf_casos", "figure"),
    [Input("tabs", "active_tab"),
     Input('drop_ciudad', 'value')],
)
def render_tab_content(active_tab, c):
    """
    Este callback arma cada pestaña
    """
    
    if active_tab is not None:
        if active_tab == "cuarentena":
            
            try:
                # =============================================================================
                # Lectura de datos  
                # =============================================================================
    
                aj = pd.read_csv("ajuste.csv", encoding='ISO-8859-1')
                pron = pd.read_csv("pronostico.csv", encoding='ISO-8859-1')
     
                
                
                # =============================================================================
                # Gráfica Casos         
                # =============================================================================
                
                # c = 'Bogotá D.C.'
                x = aj[aj['Ciudad'] == c]['Fecha']
                f_y = aj[aj['Ciudad'] == c]['Casos']
                ajuste = aj[aj['Ciudad'] == c]['Ajuste Modelo']
                pred_x = pron[pron['Ciudad'] == c]['Fecha']
                pronostico = pron[pron['Ciudad'] == c]['Pronóstico Modelo']
                l_s = pron[pron['Ciudad'] == c]['Límite Superior']
                l_i = pron[pron['Ciudad'] == c]['Límite Inferior']
                
                
                fig = go.Figure()
                # serie original
                fig.add_trace(go.Scatter(x=x, y=f_y,
                    name='Casos Reales Diarios', 
                    fill=None,
                    mode='lines',
                    line=dict(width=3)
                    ))
                # ajuste
                fig.add_trace(go.Scatter(x=x, y=ajuste,
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
                fig.add_trace(go.Scatter(x=pred_x, y=pronostico,
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
                
                dcc.Graph(figure=fig)
                return fig
            except:
                pass  
            
            
        elif active_tab == "histogram":
            
            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=data["hist_1"]), width=6),
                    dbc.Col(dcc.Graph(figure=data["hist_2"]), width=6),
                ]
            )

    return {}


@app.callback(Output("boton_actualizar", "children"), [Input("boton_actualizar", "n_clicks")])
def actualizar(n):
    
    """
    Este Callback llama la API, actualiza los modelos y guarda el resultado en un csv.
    """
    if n != None:

        df_pruebas, df_casos = data(10000000)
        df_pob = poblacion('poblacion_municipios.xlsx')
        
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
            dff = pd.merge(dff, df_recf[['Salidas']], left_index=True, right_index=True)
            dff['Activos'] = dff['Total'] - dff['Salidas']
            del df_rec, df_recf
            dff['GR'] = dff['Total'].pct_change()
            return dff
    
        
    
        pred = 30
        resul_aj = []
        resul_pron = []
        for c in ciudades:
          
            df = tabla_ciudad(c)
            
            def gompertz(x, a, b, c):
                return c * np.exp(-b * np.exp(-x / a))
            def f_gompertz(x, a, b, c):
                return a * (-b) * (-c) * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x)
            
            
            x = np.arange(0, df.shape[0])
            pred_x = np.arange(df.shape[0], df.shape[0] + pred)
            y = np.array(df['Total'])
            f_y = np.array(df[c])
            
            param, pcov = curve_fit(gompertz, x, y, maxfev=10000)
            f_param, f_pcov = curve_fit(f_gompertz, x, f_y, maxfev=10000)
            
            
            # # =============================================================================
            # # Residuales modelo original
            # # =============================================================================
            
            import scipy.stats as st
            
            res = f_y - f_gompertz(x, *f_param)
    
            # =============================================================================
            # Modelo ARMA para los errores
            # =============================================================================
            
        
            # Se prueban todos los modelos hasta p, q = 3, sin diferenciar y sin parámetros estacionales
            stepwise_fit = auto_arima(res, start_p = 1, start_q = 1, 
                                      max_p = 20, max_q = 20, 
                                      seasonal = False, 
                                      d = None, D = None, trace = False, 
                                      error_action ='ignore',   # we don't want to know if an order does not work 
                                      suppress_warnings = True,  # we don't want convergence warnings 
                                      stepwise = True,
                                      information_criterion = 'aic')           # set to stepwise 
            
            # resumen modelo
            stepwise_fit.summary()
            
            # se ajusta el modelo
            from statsmodels.tsa.statespace.sarimax import SARIMAX 
            arma = SARIMAX(res,  
                        order = stepwise_fit.order)
            
            
            resultado = arma.fit() 
            # resultado.summary() 
            
            ajuste_arma = resultado.predict(min(x), max(x), 
                                      typ = 'levels')
            
            # se junta el modelo global con el modelo arma
            ajuste = f_gompertz(x, *f_param) + ajuste_arma
    
            
            # pronóstico
              
            pron_arma = resultado.predict(min(pred_x), max(pred_x), 
                                      typ = 'levels')
            
            pronostico = f_gompertz(pred_x, *f_param) + pron_arma
            
            n_res = f_y - ajuste
            s = np.std(n_res)
            # límite superior e inferior de los intervalos
            l_s = pronostico + st.norm.ppf(.95) * s
            l_i = pronostico - st.norm.ppf(.95) * s
            
            ciud = np.repeat(c,len(ajuste))
            
            resul_aj = resul_aj + [[a,b,c,d] for a,b,c,d in zip(ciud,x,f_y,ajuste)]
            resul_pron = resul_pron + [[a,b,c,d,e] for a,b,c,d,e in zip(ciud,pred_x,pronostico,l_s,l_i)]
            
    
        df_ajuste = pd.DataFrame(data= resul_aj, columns=['Ciudad','Fecha','Casos', 'Ajuste Modelo'])
        df_pronostico = pd.DataFrame(data= resul_pron, columns=['Ciudad','Fecha','Pronóstico Modelo', 'Límite Superior','Límite Inferior'])
        
        df_ajuste.to_csv('ajuste.csv', index=False, encoding='ISO-8859-1')
        df_pronostico.to_csv('pronostico.csv', index=False, encoding='ISO-8859-1')
        
        return f"Dashboard Actualizado al {datetime.today().strftime('%d/%m/%Y')}"

    
    else: 
        return "Click para Actualizar Modelos"
    
   


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run_server(host='0.0.0.0',debug=True, port=port)
