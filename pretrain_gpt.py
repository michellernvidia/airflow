import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable

'''Python file containing functions relevant to training a gpt model from scratch and then
performing p-tuning on it'''

def download_pile_dataset(ti, org, ace, team=None):
     return


def train_gpt_model(ti, org, ace, team=None):
     return