
import requests
from urllib.parse import unquote
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import csv
import json
from operator import itemgetter
import metadata_parser
import pandas as pd
from flask import Flask, request, render_template, session, redirect
from flask_table import Table, Col
from IPython.display import HTML
from flask_table import Table, Col, LinkCol
from flask import Flask, Markup, request, url_for
from flask_paginate import Pagination, get_page_args
from flask_table import Table, Col, LinkCol
from flask import Flask, Markup, request, url_for
from bs4 import BeautifulSoup


from flask import Flask
def data():
    with requests.Session() as req:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0'}
        r = req.get(
            "https://www.barchart.com/stocks/quotes/TSLA/options?moneyness=allRows&expiration=2020-03-20", headers=headers)
        headers["X-XSRF-TOKEN"] = unquote(unquote(r.cookies.items()[0][1]))
        headers["Accept"] = "application/json"
        r = req.get("https://www.barchart.com/proxies/core-api/v1/options/get?fields=symbol%2CbaseSymbol%2CbaseLastPrice%2CbaseSymbolType%2CsymbolType%2CstrikePrice%2CexpirationDate%2CdaysToExpiration%2CbsymbolPrice%2Cmsymbolpoint%2CaskPrice%2ClastPrice%2Cvolume%2CopenInterest%2CvolumeOpenInterestRatio%2Cvolatility%2CtradeTime%2CsymbolCode&meta=field.shortName%2Cfield.type%2Cfield.description&orderBy=volumeOpenInterestRatio&orderDir=desc&baseSymbolTypes=stock&between(volumeOpenInterestRatio%2C1.24%2C)=&between(lastPrice%2C.10%2C)=&between(tradeTime%2C2022-01-01%2C2022-15-1)=&between(volume%2C50%2C)=&between(openInterest%2C10%2C)=&in(exchange%2C(AMEX%2CNASDAQ%2CNYSE))=&page=1&limit=2500", headers=headers).json()
        #print(r.keys())
        #site_json=json.loads(r)
        #print([d.get('entrezgene') for d in site_json['hits'] if d.get('entrezgene')])
        #print(r)
        #exit()
        #print(list(map(itemgetter(0), r.items())) )
        #for meta in r.items():
        #print(meta)
        # data = json.load(open('/Users/laxmanjeergal/Desktop/json.json'))
        #keys = r.keys() A
        #value = r.values()

        #print ("keys : ", str(keys))
        #print ("values : ", str(values))
        df = pd.DataFrame(r["data"])
        df = df.reset_index()
        df = df.rename(columns = {'volume':'id'})
        df = df.rename(columns = {'symbol':'name'})
        df = df.rename(columns = {'baseSymbol':'stock'})
        df = df.rename(columns = {'daysToExpiration':'expire'})
        df = df.rename(columns = {'expirationDate':'expireDate'})
        df = df.rename(columns = {'volumeOpenInterestRatio':'volOpInterest'})
        df = df.rename(columns = {'baseLastPrice':'LastPrice'})
        del df['baseSymbolType']
        del df['index']
        df = df.replace(',','', regex=True)
        #df['id'].astype(int)
        #df['expire'].astype(int)
        #df['symbolType'].astype(str)
        #df['stock'].astype(str)
        #df.sort_values('id', ascending=False).nlargest(10, 'id')
        df["id"] = df["id"].astype(str).astype(int)
        df["expire"] = df["expire"].astype(str).astype(int)
        df = df.sort_values(by=['id','expire'], ascending=False)
        #df.idxmax(axis = 0)
        #df = df.max().sort_values(ascending=True)
        #df.drop(df.columns[[5,6,7,8,9,10,11,12,13,14,15]], axis = 1, inplace = True)
        #df = pd.DataFrame(list(r.items()),columns = ['data1','meta1'])
        ##print (df.str.split(' '))
        #print(sorted_df)
        #print(final_df)
        cols = list(df.columns.values)
        #print(cols)
        #['name', 'description', 'LastPrice', 'symbolType', 'strikePrice', 'expireDate', 'expire',
        #'askPrice', 'lastPrice', 'id', 'openInterest', 'volOpInterest', 'volatility', 'tradeTime']
        df = df[['stock','symbolType', 'strikePrice', 'expire','id','openInterest','name',  'LastPrice',   'expireDate',
                 'askPrice', 'lastPrice',  'volOpInterest', 'volatility', 'tradeTime']]
        cols = list(df.columns.values)
        #print(cols)
        return df

"""
A example for creating a Table that is sortable by its header
"""

app = Flask(__name__)



app = dash.Dash(__name__)

df = data()
PAGE_SIZE = 10


def style_row_by_top_values(df, nlargest=1):
    numeric_columns = df.select_dtypes('number').drop(['id'], axis=1).columns
    styles = []
    for i in range(len(df)):
        row = df.loc[i, numeric_columns].sort_values(ascending=False)
        for j in range(nlargest):
            styles.append({
                'if': {
                    'filter_query': '{symbolType} = "Put"',
                    'column_id': 'symbolType'

                },
                'backgroundColor': 'Red',
                'color': 'white'
            })
    return styles


app.layout = html.Div(
    className="row",
    children=[
        html.Div(
            dash_table.DataTable(
                id='table-paging-with-graph',
                columns=[
                    #{"name": i, "id": i} for i in sorted(df.columns)
                    {"name": i, "id": i} for i in (df.columns)
                ],
                style_data_conditional=style_row_by_top_values(df),
                page_current=0,
                page_size=20,
                page_action='custom',

                filter_action='custom',
                filter_query='',

                sort_action='custom',
                sort_mode='multi',
                sort_by=[]
            ),
            style={'height': 500, 'overflowY': 'scroll'},
            className='six columns'
        ),



        html.Div(
            id='table-paging-with-graph-container',
            className="five columns"
        )
    ]
)





operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]



def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    Output('table-paging-with-graph', "data"),
    [Input('table-paging-with-graph', "page_current"),
     Input('table-paging-with-graph', "page_size"),
     Input('table-paging-with-graph', "sort_by"),
     Input('table-paging-with-graph', "filter_query")])
def update_table(page_current, page_size, sort_by, filter):
    filtering_expressions = filter.split(' && ')
    dff = df
    cols = list(df.columns.values)
    print("update_table")
    print(cols)

    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    return dff.iloc[
           page_current*page_size: (page_current + 1)*page_size
           ].to_dict('records')


@app.callback(


    Output('table-paging-with-graph-container', "children"),
    [Input('table-paging-with-graph', "data")])
def update_graph(rows):
    dff = pd.DataFrame(rows)
    #colors  = [dff["id"]>38865, 'red', 'green']
    dff["Color"] = np.where(dff["symbolType"] =="Put", 'red', 'green')

    return html.Div(
        [
            dcc.Graph(
                id=column,
                figure={
                    "data": [
                        {

                            "x": dff["name"] ,
                            "y": dff[column] if column in  dff else [],

                            "type": "bar",
                            "marker": {"color": dff['Color']},
                        }
                    ],

                    "layout": {
                        "xaxis": {"automargin": True},
                        "yaxis": {"automargin": True},
                        "height": 250,
                        "margin": {"t": 10, "l": 10, "r": 10},
                    },
                },
            )
            for column in ["id"]
        ]


    )


if __name__ == '__main__':
    app.run_server(debug=True)
