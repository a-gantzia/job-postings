import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

postings = pd.read_csv("data/clean_job_postings_w_salary.csv").drop(columns = ["Unnamed: 0"])
skills_lang = postings[['skills', 'programming_languages', 'location']]

def augment_skills_location(row):
    eval_lang = eval(row['programming_languages'])#.apply(lambda x: eval(x))
    eval_skills = eval(row['skills'])#.apply(lambda x: eval(x))
    lang_str = ' '.join(eval_lang)
    skills_str = ' '.join(eval_skills)
    return lang_str + ' ' + skills_str + " " + row['location'].lower()

skills_lang['joined_skills'] = skills_lang.apply(augment_skills_location, axis = 1)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(skills_lang['joined_skills'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(skills, locations, similarities, jobs):
    skills_str = ' '.join(input_skills)
    locations_str = ' '.join(locations)
    desired_str = skills_str + " " + locations_str
    
    vectorized = tfidf_vectorizer.transform([desired_str])
    sim_score = cosine_similarity(vectorized, tfidf_matrix)[0]
    most_similar_jobs = sim_score.argsort()[::-1][:5]
    recommended_job = jobs.iloc[most_similar_jobs][['clean_job_title', 'location', 'clean_job_description', 
                                                    'salary estimate', 'rating']]
    return recommended_job

input_skills = ['python', 'java', 'R', 'Operations']
locations = ['Harvard']
recommendation = recommend(input_skills, locations, cosine_sim, postings)
