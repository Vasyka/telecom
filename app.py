import re
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from scipy.sparse import save_npz, load_npz

import dash
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc


# Load data
df = pd.read_csv('data/Telecom_with_topics.csv',index_col=0)

# Load lda model,topics and keywords
top_words = pd.read_csv('data/top_words.csv',index_col=0)
with open('data/lda.pkl', 'rb') as f:
    lda = pickle.load(f)

# Load vectorizer and vectors
def lemma_tokenize(text):
    return ['']
with open('data/count_vectorizer.pkl', 'rb') as fi:
    vectorizer = pickle.load(fi)
text_tfidf = load_npz('data/X_tfidf.npz')

# Get scores for topics for all texts
predicted = lda.transform(text_tfidf)
top_scores = np.max(predicted,axis=1)

# Add scores for topics to data table
text_lens = df.Text.apply(lambda text: len(re.split(r'[\s,.!]',text)))
df.loc[:,'top_score'] = np.NaN
df.loc[text_lens>=4,'top_score'] = top_scores

# Load top words and reorder index
top_words_values = pd.read_csv('data/top_words_values.csv',index_col=0)
top_words_values = top_words_values.loc[top_words_values.index[::-1]]
top_words = top_words.loc[top_words.index[::-1]]

sub_df = df.loc[:4,:]


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

table_style = {
        'whiteSpace': 'normal',
        'height': 'auto',
        'overflowY': 'scroll',
        'border': '1px solid darkgrey'
    }

description = dbc.Card( 
                html.P('''Комментарий: cправа показан топ ключевых слов для категории. 
                При этом это уже восстановленные слова к тому виду, какими они были до препроцессинга. 
                Тем не менее англо-язычные сокращения вроде \'pc\' не восстанавливались, 
                поскольку они являются частями названий услуг или подписок и идентифицировать их значение мы все  
                равно не можем без дополнительных данных. Также часть ключевых слов имеет смысл только вместе 
                друг с другом и поэтому для наглядности они были заменены на словосочетания. Такие слова заметно выделяются 
                на графике и для них показан скор как суммарный, так и по отдельности для каждого слова.'''),body=True)

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Оператор:'),
                dcc.Checklist(
                    id='operator_checkbox',
                    options=[
                        {'label': 'Оператор1', 'value': 'Оператор1'},
                        {'label': 'Оператор2', 'value': 'Оператор2'},
                        {'label': 'Оператор3', 'value': 'Оператор3'}
                    ],
                    value=['Оператор1','Оператор2','Оператор3'],
                    inputStyle={"margin-right": "5px"},
                    labelStyle={"margin-right": "10px"},
                ),
                
            ]),
        dbc.FormGroup(
            [
               dbc.Label('Тип услуги оказываемой оператором:'), 
               dcc.Checklist(
                    id='type_checkbox',
                    options=[
                        {'label': 'Интернет', 'value': 'Интернет'},
                        {'label': 'Мобильная связь', 'value': 'Мобильная связь'}
                    ],
                    value=['Интернет','Мобильная связь'],
                    inputStyle={"margin-right": "5px"},
                    labelStyle={"margin-right": "10px"},
                ),
            ])
    ],body=True,
    
)
app.layout = dbc.Container(
    [
        html.H1("HSE: Data Science в клиентской и текстовой аналитике: Домашнее задание 2 (текстовая аналитика)"),
        html.Hr(),
        html.H5('По умолчанию показывается стастистика по всем операторам и типам услуг. Кликните \
                по столбцу на гистограмме, соответствующему интересующей теме обращений. Справа появится \
                топ 15 слов для данной темы. Промотайте страницу ниже, чтобы посмотреть на тексты обращений для \
                выбранной темы. Также вы можете выбрать оператора и тип услуги оказываемой оператором.'),
        dbc.Row([  
            dbc.Col(controls, md=4),
            dbc.Col(description, md = 8)
        ]),
        dbc.Row(
            [
                
                dbc.Col(dcc.Graph(id="topic_hist", 
                                  figure={'data':[], 
                                          'layout':{
                                              'height':900,
                                          }}), md=7),
                dbc.Col(dcc.Graph(id="top_words",
                                  figure={'data':[], 
                                          'layout':{
                                              'height':900,
                                          }}), md=5),
            ],
            align="center",
        ),
        
        dbc.Row([html.H5(id='topic')],
                justify="center",align="center"),
        
        dash_table.DataTable(
            id='data_table',
            style_data = table_style,
            style_cell={'textAlign': 'left'},
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Text'},
                    'size': '700px'
                }
            ],
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold',
                'border': '1px solid black'
            },
            columns=[{"name": i, "id": i} for i in sub_df.columns],
            data=sub_df.to_dict('records'),
        ),
        
    ],
    fluid=True,
)


# Callback for histogram
@app.callback(
    Output("topic_hist", "figure"), 
    [Input("operator_checkbox", "value"), 
     Input("type_checkbox", "value")])
def generate_chart(operators, types):
    fig ={'data':[]}
    if len(operators)>0 and len(types)>0: 
        part_df = df[df.Operator.isin(operators)&df.Type.isin(types)]
        fig = px.histogram(part_df, x="topic", log_y=True, nbins=15,
                          color = 'Type') 
                           
                           
        fig.update_layout(title_text='Распределение обращений по темам(log-scale)', title_x=0.5)
    
    return fig

# Callback for top words
@app.callback(
    Output('top_words', 'figure'),
    [Input('topic_hist', 'clickData')])
def display_click_topic_words(clickData):
    fig ={'data':[]}
    if clickData is not None:
        topic = clickData['points'][0]['x']
        title = 'Топ 15 слов для темы ' + topic
        if topic != 'Прочее':
            fig = px.bar(x=top_words_values[topic], y = top_words[topic], 
                         orientation='h',color=top_words_values[topic])
            fig.update_layout(title_text=title, title_x=0.5)
        else:
            fig ={'data':[],'layout': {'title_text':title, 
                                       'title_x':0.5}}
    return fig

# Callback for header above table
@app.callback(
    Output('topic', 'children'),
    [Input('topic_hist', 'clickData')])
def display_click_topic(clickData):
    content = "Показываются первые 3 строки таблицы. Кликните на столбец на гистограмме, чтобы увидеть \
                тексты для конкретного топика"
    if clickData is not None:
        topic = clickData['points'][0]['x']
        content = "Тема: " + topic + ". Показываются случайные 10 обращений."
    return content

# Callback for table
@app.callback(
    Output('data_table', 'data'),
    [Input('topic_hist', 'clickData'),
     Input("operator_checkbox", "value"),
     Input("type_checkbox", "value")])
def display_click_data(clickData, operators, types):
    if clickData is not None:
        if len(operators)>0 and len(types)>0: 
            part_df = df[df.Operator.isin(operators)&df.Type.isin(types)]
            topic = clickData['points'][0]['x']
            df_topic = part_df[part_df.topic==topic]
            if len(df_topic) > 10:
                df_topic = df_topic.sample(10)
            df_topic = df_topic.sort_values(by='top_score',ascending=False)
            return df_topic.to_dict('records')
        
    return []


if __name__ == '__main__':
    app.run_server(debug=True)
