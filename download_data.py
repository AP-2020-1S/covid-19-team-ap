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