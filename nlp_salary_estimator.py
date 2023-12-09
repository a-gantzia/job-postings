import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import pipeline
import torch
from salary_estimator import salary_range
import requests


def get_highest_label_number(label_data):
    highest_label_number = -1

    for item in label_data:
        # Extract the number part from the label string
        label_number = int(item["label"].split("_")[1])

        # Update the highest label number if the current one is greater
        if label_number > highest_label_number:
            highest_label_number = label_number

    return highest_label_number


def predict_salary_range_desc(description, API_KEY):
    API_URL = (
        "https://api-inference.huggingface.co/models/benjaminrio/job-salary-classifier"
    )
    headers = {"Authorization": f"Bearer {API_KEY}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query(
        {
            "inputs": description,
        }
    )
    prediction = get_highest_label_number(output[0])
    return salary_range(prediction)
