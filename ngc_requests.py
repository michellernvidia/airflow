import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable


def get_token(ti, key, org=None, team=None):
    '''Use the api key set environment variable to generate auth token'''
    scope_list = []
    scope = f'group/ngc:{org}'
    scope_list.append(scope)
    if team:
        team_scope = f'group/ngc:{org}/{team}'
        scope_list.append(team_scope)

    querystring = {"service": "ngc", "scope": scope_list}

    auth = '$oauthtoken:{0}'.format(key)
    auth = base64.b64encode(auth.encode('utf-8')).decode('utf-8')
    headers = {
        'Authorization': f'Basic {auth}',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
    }
    url = 'https://authn.nvidia.com/token'
    response = requests.request("GET", url, headers=headers, params=querystring)
    if response.status_code != 200:
        raise Exception("HTTP Error %d: from %s" % (response.status_code, url))
    return json.loads(response.text.encode('utf8'))["token"]


# 2. Download a NeMo checkpoint into a Workspace 
def create_workspace(ti, org, ace, workspace_name):
        token = ti.xcom_pull(task_ids='token')
        print(f"Xcom pull gives me {token}")
        
        '''Create a workspace in a given org for the authenticated user'''
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/workspaces/'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
         }
        
        # nemo_ckpt_.split('.')[0] + 
        data = {
          'aceName': f'{ace}',
          'name': workspace_name
         }
        response = requests.request("POST", url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            print(response)
            raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
        
        return response.json()


def ngc_job_request(ti, org, job_name, ace_instance, ace_name, docker_image, replica_count, \
                    workspace_mount_path, workspace_id, job_command, team=None):
      '''Creates an NGC job request via API'''
      token = ti.xcom_pull(task_ids='token')
      if team:
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/jobs/'
      else:
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/'

      headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
      data = {
                "name": job_name,
                "aceInstance": ace_instance,
                "aceName": ace_name,
                "dockerImageName": docker_image,
                "jobOrder": 50,
                "jobPriority": "NORMAL",
                "replicaCount": replica_count,
                "reservedLabels": [],
                "resultContainerMountPoint": "/results",
                "runPolicy": {
                    "preemptClass": "RUNONCE"
                },
                "systemLabels": [],
                "userLabels": [],
                "userSecretsSpec": [],
                "workspaceMounts": [
                    {
                        "containerMountPoint": workspace_mount_path,
                        "id": workspace_id,
                        "mountMode": "RW"
                    }
                ],
                "command": job_command
            } 
      response = requests.request("POST", url, headers=headers, data=json.dumps(data))
      if response.status_code != 200:
            raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
      return response.json()


def ngc_job_status(ti, org, job_id):
    '''Gets status of NGC Job (e.g., SUCCESS, FAILED, CREATED, etc.)'''
    
    token = ti.xcom_pull(task_ids='token')
    url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/{job_id}'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.request("GET", url, headers=headers)
    if response.status_code != 200:
        raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
    job_info = response.json()
    return job_info['job']['jobStatus']['status']