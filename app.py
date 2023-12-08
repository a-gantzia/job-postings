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
from plots_package import plot_maker
import collections
import string
from  PIL import Image
import requests
from pathlib import Path
from salary_estimator import predict_salary_range


from io import BytesIO
import collections
import urllib.request
from wordcloud import WordCloud, ImageColorGenerator


import nltk 
st.set_option('deprecation.showPyplotGlobalUse', False)


data_filepath = 'data/'
df = pd.read_csv(data_filepath+'clean_job_postings_w_salary.csv')
salary_COL = pd.read_csv(data_filepath+"salary_cost_of_living.csv")

sns.set_style("whitegrid", {'axes.grid' : False})

jobs=['Data Scientist', 'Data Analyst', 'Data Engineer', 'Machine Learning Engineer']

with st.sidebar:
    choose = option_menu("Main Menu", ["About", 
                                       "Analysing Job Market", 
                                       "Salary Estimator",
                                       "Job Recommender", 
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
    
if choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:     
        st.markdown(style1, unsafe_allow_html=True)
        st.markdown('<p class="font">Your Personalized Path to a Fair Salary</p>', unsafe_allow_html=True)    
    st.write("This demo is meant to allow job applicants in Tech tailored to you.")
    st.write("We are using", len(df), "different US tech job positions between 2022 and 2023, which have been used to train our Salary Estimator.")
    st.write("""Are you navigating the tech job market? Step into a future where you're empowered with knowledge and insight. Our cutting-edge demo is not just a tool; it's your personal guide to the tech industry's salary landscape.

What We Offer:
- Customized Job Matches: Engage with our interactive demo designed specifically for tech job seekers like you. We understand that every applicant is unique, so we've tailored our technology to align with your skills, experience, and career aspirations.
- Data-Driven Salary Estimator: Leverage the power of data in the form of US tech job positings. Our Salary Estimator isn't just numbers—it's a compass that guides you to a salary that reflects your worth.

Why Choose Our Demo?
- Insightful Analytics: Go beyond the basic job search. Our demo offers a deep dive into salary trends, giving you a competitive edge in salary negotiations.
- Empowerment in Your Job Search: With our Salary Estimator, knowledge is power. Understand the financial implications of your next career move before you make it.
- Stay Ahead of the Curve: The tech industry evolves rapidly. So does our data. Keep pace with the latest salary trends and job positions to ensure you're always one step ahead.""")

elif choose=='Analysing Job Market':
    st.markdown(style1, unsafe_allow_html=True)
    st.markdown('<p class="font">Analysing the Job Market</p>', unsafe_allow_html=True)
    selected_job = st.selectbox("Select your ideal job position", 
                                    jobs, index = 0).lower()
    if selected_job:
        salary_COL=salary_COL[salary_COL.job_category==selected_job]
        df = df[df.job_category==selected_job]
        st.write("According to", (sum(salary_COL.observations)), """different US tech job positions between in this occupation, we provide an analysis of the Skills you need and the  expectations you should have 
            according to the US State you could work from (accounting also for cost of living).""")
        
        st.markdown(style2, unsafe_allow_html=True)
        st.markdown('<p class="font">Salary and Cost of Living Expectations</p>', unsafe_allow_html=True)
        fig1 = plot_maker.create_salary_map(salary_COL)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = plot_maker.create_salary_COL_plot(salary_COL)
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = plot_maker.create_seniority_plot(df)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<p class="font">Skill Set Required</p>', unsafe_allow_html=True)
        fig4 = plot_maker.create_languages_plot(df)
        st.plotly_chart(fig4, use_container_width=True)
        ppl_standing_img = 'https://t3.ftcdn.net/jpg/03/95/34/52/360_F_395345213_HoE6wTdwF6vEagoOqNVkkuN3WmRBFROM.jpg'
        fig5 = plot_maker.makingclouds(df,'clean_job_description_filtered',ppl_standing_img, 'wordcloud_ppl_standing')
        plt.axis("off")
        st.pyplot()

elif choose == "Contact":
    st.markdown(style1, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Us!</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True):
        name=st.text_input(label='Please Enter Your Name')
        email=st.text_input(label='Please Enter Email')
        message=st.text_input(label='Please Enter Your Message')
        submitted = st.form_submit_button('Submit')
        if submitted:
            slack_notify(name,email,message)
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')
        
elif choose=='Salary Estimator':
    st.markdown(style1, unsafe_allow_html=True)
    st.markdown('<p class="font">Salary Estimator</p>', unsafe_allow_html=True)
    input_data = {}

    input_data['skills'] = st.text_input("Enter skills (comma-separated):")
    input_data['prog_lang'] = st.text_input("Enter programming languages (comma-separated):")
    input_data['seniority'] = st.selectbox("Select seniority level", ["Junior", "Mid", "Senior"], index=0)
    categories = ['location', 'company_type', 'job_category', 'company_sector', 'company_industry']
    for category in categories: 
        with open(category + '.pkl', "rb") as fd:
            temp = pickle.load(fd)
        input_data[category] = st.selectbox("Select " +  category, temp, index=0)
    input_data["size"] = st.selectbox("Select company size", ['1 to 50 Employees',
                                                            '51 to 200 Employees',
                                                            '201 to 500 Employees',
                                                            '501 to 1000 Employees',
                                                            '1001 to 5000 Employees',
                                                            '5001 to 10000 Employees',
                                                            '10000+ Employees',
                                                            'Unknown'], index = 0)
    

    predicted_range = predict_salary_range(input_data)
    st.markdown(style1, unsafe_allow_html=True)
    st.markdown(f'<p class="font">{predicted_range}</p>', unsafe_allow_html=True)



    
    
    # col1, col2 = st.columns( [0.5, 0.5])
    # with col1:

elif choose=='Job Recommender':
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    input_skills = st.text_input("Enter skills (comma-separated):")
    selected_skills = [skill.strip() for skill in input_skills.split(',')]
    selected_areas = st.text_input("Input two areas of interest where you would like to work (comma-separated)")
    selected_areas = [area.strip() for area in selected_areas.split(',')]

    postings = pd.read_csv("data/clean_job_postings_w_salary.csv").drop(columns = ["Unnamed: 0"])
    skills_lang = postings[['skills', 'programming_languages', 'location']]

    def augment_skills_location(row):
        eval_lang = eval(row['programming_languages'])
        eval_skills = eval(row['skills'])
        lang_str = ' '.join(eval_lang)
        skills_str = ' '.join(eval_skills)
        return lang_str + ' ' + skills_str + " " + row['location'].lower()

    skills_lang['joined_skills'] = skills_lang.apply(augment_skills_location, axis = 1)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(skills_lang['joined_skills'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    from recommender_system import recommend

    df = recommend(selected_skills, selected_areas, cosine_sim, postings)
    df.columns = ["Job Title", "Location", "Description", "Estimated Salary", "Rating"]
    df = df.reset_index().drop(columns = ['index'])
    st.header("Here is a table of recommended jobs that you should check out based on your skills and areas of interest:")
    st.dataframe(df)





































