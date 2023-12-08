import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def salary_range(i):
    if i == 0:
        return "10,000 - 57,844"
    if i == 1:
        return "57,844 - 105,675"
    if i == 2:
        return "105,675 - 153,506"
    if i == 3: 
        return "153,506 - 201,337"
    if i == 4:
        return "201,337 - 249168"
    return "249168 - 297,000"


def predict_salary_range(data_input):
    df = pd.DataFrame([data_input])
    with open("columns.pkl", "rb") as fd:
        cols = pickle.load(fd)

    input_data = {}

    df["prog_langs"] = df["prog_lang"].apply(lambda x: x.strip().split(","))
    df["skills"] = df["skills"].apply(lambda x: x.strip().split(","))
    for col in cols:
        input_data[col] = 0

    if df["seniority"].values[0] == "Junior":
        input_data["seniority_encoded"] = 0
    elif df["seniority"].values[0] == "Mid":
        input_data["seniority_encoded"] = 1
    else: 
        input_data["seniority_encoded"] = 2

    input_data["location_" + df["location"].values[0]] = 1
    input_data["job_category_" + df["job_category"].values[0]] = 1
    input_data["company_sector_" + df["company_sector"].values[0]] = 1
    input_data["company_industry_" + df["company_industry"].values[0]] = 1

    size_list_reordered = [
        '1 to 50 Employees',
        '51 to 200 Employees',
        '201 to 500 Employees',
        '501 to 1000 Employees',
        '1001 to 5000 Employees',
        '5001 to 10000 Employees',
        '10000+ Employees',
        'Unknown'
    ]

    revenue_list_reordered = [
        'Less than $1 million (USD)',
        '$1 to $5 million (USD)',
        '$5 to $25 million (USD)',
        '$25 to $100 million (USD)',
        '$100 to $500 million (USD)',
        '$500 million to $1 billion (USD)',
        '$1 to $5 billion (USD)',
        '$5 to $10 billion (USD)',
        '$10+ billion (USD)',
        'Unknown / Non-Applicable'
    ]


    for i, size in enumerate(size_list_reordered):
        if df["size"].values[0] == size:
            input_data["company_size_encoded"] = i

    for i, size in enumerate(revenue_list_reordered):
        if df["size"].values[0] == size:
            input_data["company_revenue_encoded"] = i

    for lang in df["prog_langs"].values[0]:
        input_data[lang] = 1

    for skill in df["skills"].values[0]:
        input_data[skill] = 1

    input_data['num_of_programming_languages'] = len(df["prog_langs"].values[0])
    input_data['num_of_skills'] = len(df["skills"].values[0])


    val = pd.DataFrame([input_data])


    with open("Salary_classifier.pkl", "rb") as fd:
        model = pickle.load(fd)
    d_test = xgb.DMatrix(val)
    pred = model.predict(d_test)
    return salary_range(pred)





