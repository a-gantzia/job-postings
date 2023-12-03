import re
from re import sub
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from io import BytesIO
import collections
from PIL import Image
import urllib.request
from wordcloud import WordCloud, ImageColorGenerator

import nltk 

import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import seaborn as sns
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

import re
from re import sub
import pickle

from io import BytesIO
import collections
from PIL import Image
import urllib.request
from wordcloud import WordCloud, ImageColorGenerator
import string

import urllib
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')
blue_to_red = [
    [0, 'rgb(0, 0, 255)'],
    [1, 'rgb(255, 0, 0)'] 
]


def create_salary_map(df):
    size_scale = 10 

  
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.8, 0.2],
        specs=[[{'type': 'scattergeo'}, {'type': 'bar'}]]
    )

    fig.add_trace(
        go.Scattergeo(
            locationmode='USA-states',
            locations=df['state'], 
            text=df['hovertext'],
            marker=dict(
                size=df['observations'] * size_scale, 
                color=df['average_salary'],
                colorscale=blue_to_red,  
                colorbar_title='Average Salary',
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode='area',
                cmin=np.min(df['average_salary']),
                cmax=np.max(df['average_salary']),
                colorbar=dict(x=-0.1)
            ),
            hoverinfo='text'
        ),
        row=1, col=1
    )

    remote_jobs_index = df['state'] == 'Remote'
    mean_salary_remote = df['average_salary'][remote_jobs_index].mean()

    number_of_remote_jobs = df['observations'][remote_jobs_index].sum()

    bar_color = (mean_salary_remote - np.min(df['average_salary'])) / (np.max(df['average_salary']) - np.min(df['average_salary']))

    fig.add_trace(
        go.Bar(
            x=['Remote Jobs'],
            y=[number_of_remote_jobs],
            #marker=0.7,#dict(color=bar_color),
            marker=dict( 
                color='rgb(180,0,0)',
                colorscale=blue_to_red,
            ),
            text=f'Average Salary: ${mean_salary_remote:,.2f}',
            hovertext=f'Average Salary: ${mean_salary_remote:,.2f}, based on {number_of_remote_jobs} jobs', 
            hoverinfo='text',
            textposition='auto'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='Average Salary by State in the USA and Remote Jobs',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            domain=dict(x=[0, 0.75]),  
        ),
        showlegend=False
    )

    fig.update_layout(
        barmode='stack',
        annotations=[dict(
            x=1.25,
            y=number_of_remote_jobs,
            text='Remote Jobs',
            showarrow=False,
            xref='paper',
            yref='y',
            font=dict(size=16)
        )],
        yaxis2=dict(showticklabels=False)
    )

    fig.update_layout(
        width=1000,
        height=600
    )

    return fig

def create_salary_COL_plot(df):
    df = df.sort_values('median_salary_after_COL', ascending=False)
    df=df[df.state!='Remote']
    df=df[df.observations>=4]
    fig = px.bar(df,
                 x='state',
                 y='median_salary_after_COL',
                 title='Pre-Tax Average Income After Cost of Living Expenses in US States',
                 labels={'median_salary_after_COL': 'Median Salary', 'Unfiltered_State': 'State'},
                 color='median_salary_after_COL',color_continuous_scale="Bluered"
                 )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_seniority_plot(df):
    pivot_df = pd.pivot_table(df,values='salary',index=['job_category','seniority'])
    fig = px.bar(pivot_df.reset_index(), x='job_category', y='salary', color='seniority', color_discrete_sequence=["red", "blue", "goldenrod"], 
                 title='Salary by Job Category and Seniority')

    fig.update_layout(xaxis_title='Job Category', yaxis_title='Salary', barmode='group')
    return fig

def create_languages_plot(df):
    df['programming_languages']= [eval(lst) for lst in df['programming_languages']]
    lang_df = df['programming_languages'].explode().value_counts().reset_index()
    lang_df.columns = ['Languages', "jobs"]
    fig = px.bar(lang_df[:8], 
             x="Languages", 
             y="jobs", 
             color = "jobs",color_continuous_scale="Bluered"
             )
    return fig


def makingclouds(frame,col,img_link, title):
    cloudtext = ' '.join(map(str, frame[col]))
    word_freq = nltk.FreqDist([i for i in cloudtext.split() if len(i) > 2])
    mask, img_color = apply_image_mask(img_link)
    wc = WordCloud(background_color='white',
                   max_font_size=100,
                   max_words=500,
                   mask = mask,
                   random_state=42)
    wordcloud = wc.generate_from_frequencies(word_freq)
    wordcloud = wordcloud.recolor(color_func=img_color)
    plt.axis("off")
    plt.figure()
    plt.axis("off")
    fig = plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig


def apply_image_mask(img_link):
    with urllib.request.urlopen(img_link) as url:
        f = BytesIO(url.read())
    img = Image.open(f)
    mask = np.array(img)
    img_color = ImageColorGenerator(mask)
    return mask, img_color


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 16
LARGE_SIZE = 24

params = {
    'figure.figsize': (16, 8),
    'font.size': SMALL_SIZE,
    'xtick.labelsize': MEDIUM_SIZE,
    'ytick.labelsize': MEDIUM_SIZE,
    'legend.fontsize': BIG_SIZE,
    'figure.titlesize': LARGE_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': BIG_SIZE
}
plt.rcParams.update(params)














