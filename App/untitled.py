import sys
sys.path.append( '..' )

import streamlit as st
import streamlit.components.v1 as html
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import collections
import string

from  PIL import Image
import requests
import json

data_filepath = '..\\data\\'
pic_filepath = '..\\pics\\'
with open(data_filepath +'df_grouped.pkl', 'rb') as f:
    df = pickle.load(f)
    
jobs = df.index.tolist()

sns.set_style("whitegrid", {'axes.grid' : False})
st.set_page_config(page_title='Salary Estimator', 
                   layout="wide")

#st.beta_set_page_config(, page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
# side bar menu
with st.sidebar:
    choose = option_menu("Main Menu", ["About", 
                                       "Skills Requirements", 
                                       "Salary Estimator", 
                                       "Contact"],
                         icons=['house',
                                'app-indicator', 
                                'file-slides',
                                'person lines fill'],
                         menu_icon="list", default_index=0,
                         styles={"container": {"padding": "5!important", 
                                               "background-color": "#fffff"},
                                 "icon": {"color": "#3385ff", 
                                          "font-size": "25px"}, 
                                 "nav-link": {"font-size": "16px", 
                                              "text-align": "left", 
                                              "margin":"0px", 
                                              "--hover-color": "#eee"},
                                 "nav-link-selected": {"background-color": "#adbbc7"},
                                }
                        )
    style1 = """ <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: #3355ff;} 
                </style> """
    style2 = """ <style> .font {
                font-size:25px ; font-family: 'Cooper Black'; color: #3385ff;} 
                </style> """
    
logo = Image.open(pic_filepath + 'logo.png')
if choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(style1, unsafe_allow_html=True)
        st.markdown('<p class="font">About</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130)
    st.write("Big Blue Data Academy offers hands-on Data Science training that bridges academia and industry. This app facilitates applicants of technology-related postions to up-skill. We indicate the most wanted skills according to the market and offer course suggestions tailored to you.")
    st.write("We have identified", len(df), "different tech job positions, which have been vectorised, clustered and mapped to a cartesian coordinate system on the diagram below.")
    # plotting the titles
    fig = px.scatter(df, x="Dimension 1", y="Dimension 2", 
                     color="cluster", 
                     size=[0.65]*len(df), 
                     text=df.index, 
                     height=650, 
                     width=800)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Created by Angeliki Gantzia.")
    st.image(logo, width=300)
    
    
    
elif choose=='Skills Requirements':
    st.markdown(style2, unsafe_allow_html=True)
    st.markdown('<p class="font">Watch a short demo of the app...</p>', unsafe_allow_html=True)
    not_found = Image.open(pic_filepath + 'under-construction.png')
    st.image(not_found, width=700)
    #video_file = open('Demo.mp4', 'rb')
    #video_bytes = video_file.read()
    #st.video(video_bytes)
    
    
elif choose == "Contact":
    st.markdown(style1, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Us!</p>', unsafe_allow_html=True)
    st.write("In Big Blue Data Academy, we are always happy to hear from you! You may reach us through the contact form below or direcly by e-mail at: info@bigblue.academy")
    with st.form(key='columns_in_form2',clear_on_submit=True):
        name=st.text_input(label='Please Enter Your Name')
        email=st.text_input(label='Please Enter Email')
        message=st.text_input(label='Please Enter Your Message')
        submitted = st.form_submit_button('Submit')
        if submitted:
            slack_notify(name,email,message)
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')
        
    st.image(logo, width=300)
    
elif choose=='Salary Estimator':
    st.title("Welcome to Big Blue's Career Prep!")
    st.markdown(style2, unsafe_allow_html=True)
    st.markdown('<p class="font">Come closer to your dream job, One qualification at a time...</p>', unsafe_allow_html=True)
    #st.markdown('<p class="font">One qualification at a time</p>', unsafe_allow_html=True)
    #st.markdown('<p class="font">Upload your CV</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CV to find out which skills you are missing (Optional)",
                                     type=['pdf'], key="2")
    cv = ''
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                cv += page.extract_text()
        cv = cv.lower()    
        
    selected_job = st.selectbox("What's your ideal tech career?", 
                                jobs, index = 36)
    observations = df.loc[selected_job].num_of_observ

    st.write('Based on ', observations, ' job posts, these are the top needed programming languages and skills:')
    lang, l_sc = suggestion.suggest_top_skills(df, selected_job, 'all_langs', topN=10)
    missing_langs = suggestion.suggest_missing_skills(cv, lang)   
    sugg_l = suggestion.suggest_course(courses, missing_langs)
    
    skill, s_sc = suggestion.suggest_top_skills(df, selected_job, 'all_skills', topN=10)
    missing_skills = suggestion.suggest_missing_skills(cv, skill)
    sugg_s = suggestion.suggest_course(courses, missing_skills)
    
    # Setting the visualization parameters
    fig_2 = make_subplots(rows=1, cols=2,
                      specs=[[{'type': 'xy'}, {"type": "xy"}]],
                      subplot_titles=("Top 10 Programming Languages", 
                                      "Top 10 Skills"))
    fig_2.add_trace(go.Bar(x=lang, 
                       y=l_sc,
                       name ='Language',
                       marker_color=px.colors.sequential.Plasma),
                       row=1, col=1)
    fig_2.add_trace(go.Bar(x=skill, 
                       y=s_sc,
                       name ='Skill',
                       marker_color=px.colors.sequential.Plasma),
                       row=1, col=2)
    fig_2.update_traces(marker_line_width=0)
    
    fig_2.update_layout(showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Arial', 
                              size=13))
    #fig_2.update_annotations(font=dict(family="Arial", size=20))
    
    
    fig_2.update_xaxes(tickangle=45, tickfont=dict(#color='crimson', 
                                                   size=15))
    
    
    # Displaying the plot
    st.plotly_chart(fig_2, use_container_width=True)
    
    def aggrid_interactive_table(df: pd.DataFrame, hyperlink = False):
        options = GridOptionsBuilder.from_dataframe(
            df, enableRowGroup=True, enableValue=True, enablePivot=True)
        options.configure_side_bar()
        options.configure_selection("single")
        if hyperlink:
            options.configure_column("Name Link", headerName="Name",
                                cellRenderer=JsCode('''function(params)
                                {return '<a href="' 
                                + params.value.split(',')[0]
                                + '" target="_blank">' 
                                + params.value.split(',').slice(1).join(',') 
                                + '</a>'}'''),
                                width=300)
        selection = AgGrid(df, enable_enterprise_modules=True,
                           gridOptions=options.build(),
                           fit_columns_on_grid_load=True,
                           theme='dark', 
                           update_mode=GridUpdateMode.MODEL_CHANGED,
                           allow_unsafe_jscode=True)
        return selection

    cols = ['Skill', 'Course Name','Stars','Reviews',] 
    
    if cv:
        st.write('You are missing ', len(missing_langs), 'out of 10 programming languages. These are some course suggestions for you:',  missing_langs)
    else: 
        st.write('Here are some course suggestions for the top programming languages:')
    selection_l = aggrid_interactive_table(df=sugg_l, hyperlink = True)
                 
    if cv:
        st.write('You are missing ', len(missing_skills), 'out of 10 skills and tools. These are some course suggestions for you:',  missing_skills)
    else: 
        st.write('Here are some course suggestions for the 10 top skills and tools:')
    selection_s = aggrid_interactive_table(df=sugg_s, hyperlink = True)
    
    
    st.write('Not sure this is your ideal career? Here are some similar positions!')
    similar_jobs = suggestion.suggest_jobs(df, selected_job)
    similar_jobs = pd.DataFrame(similar_jobs, columns = ['Similar Jobs'])
    similar_jobs = similar_jobs.set_index('Similar Jobs').join(df)
    similar_jobs = similar_jobs[['num_of_skills', 'num_of_lang', 'num_of_observ', 'skills', 'langs']]
    similar_jobs = similar_jobs.rename( columns = 
                                       {"num_of_skills": "Skills (Median)", 
                                        "num_of_lang": "Programming Languages (Median)",
                                        "num_of_observ": "Observations"})
    similar_jobs = similar_jobs.reset_index()
    selection = aggrid_interactive_table(df=similar_jobs)
    
    if selection:
        st.write("You selected:")
        st.json(selection["selected_rows"])

