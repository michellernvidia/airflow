import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.models import Variable

## 0. Variables
key_v = Variable.get("key_v", deserialize_json=True)
org_v = Variable.get("org_v", deserialize_json=True)
team_v = Variable.get("team_v", deserialize_json=True)
ace_v = Variable.get("ace_v", deserialize_json=True)
workspace_name_v = Variable.get("workspace_name_v", deserialize_json=True)
nemo_ckpt_v = Variable.get("nemo_ckpt_v", deserialize_json=True)

key_= str(key_v)
org_=str(org_v)
team_= str(team_v)
ace_=str(ace_v)
workspace_name_ = str(workspace_name_v)
nemo_ckpt_=str(nemo_ckpt_v)


#1. Connect to BCP API
def get_token(ti, org=None, team=None):
    '''Use the api key set environment variable to generate auth token'''
    scope_list = []
    scope = f'group/ngc:{org}'
    scope_list.append(scope)
    if team:
        team_scope = f'group/ngc:{org}/{team}'
        scope_list.append(team_scope)

    querystring = {"service": "ngc", "scope": scope_list}

    auth = '$oauthtoken:{0}'.format(key_)
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

def ngc_job_request(ti, org, data, team=None):
      '''Creates an NGC job request via API'''
      token = ti.xcom_pull(task_ids='token')
      if team:
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/jobs/'
      else:
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/'

      headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
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


def download_nemo_checkpoint(ti, org, ace, team=None):
      
      #get workspace id
      workspace_response = ti.xcom_pull(task_ids='workspace')
      workspace_id = workspace_response['workspace']['id']

      #define job specifics
      data = {
                "name": "airflow_download_gpt3_5b_ckpt",
                "aceInstance": "dgxa100.80g.1.norm",
                "aceName": ace,
                "dockerImageName": f"{org}/nemofw-training:23.05-py3",
                "jobOrder": 50,
                "jobPriority": "NORMAL",
                "replicaCount": 1,
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
                        "containerMountPoint": "/mount/data",
                        "id": workspace_id,
                        "mountMode": "RW"
                    }
                ],
                "command": "cd ../; cd /mount/data/; mkdir gpt_models; cd gpt_models;\
                    wget https://huggingface.co/nvidia/nemo-megatron-gpt-5B/resolve/main/nemo_gpt5B_bf16_tp2.nemo"
            }
      
      job_response = ngc_job_request(ti, org, data, team)
      job_id = job_response['job']['id']

      #keep waiting until job completes
      job_status = ngc_job_status(ti, org, job_id)
      while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED':
            time.sleep(20)
            job_status = ngc_job_status(ti, org, job_id)
            print(job_status)

      return job_response

# 3. Run p-tuning (training)
def p_tuning_training_bcp(ti, org, ace, team=None):
      
      #get workspace id
      workspace_response = ti.xcom_pull(task_ids='workspace')
      workspace_id = workspace_response['workspace']['id']

      #actual command to run p-tuning with NeMo framework  
      p_tuning_command = "python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
                            prompt_learning=gpt3/squad \
                            stages=[prompt_learning] \
                            cluster_type=bcp \
                            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
                            data_dir=/mount/workspace \
                            base_results_dir=/mount/workspace/results \
                            prompt_learning.run.model_train_name=gpt3_5b \
                            prompt_learning.trainer.devices=4 \
                            prompt_learning.model.language_model_path=/mount/workspace/gpt_models/nemo_gpt5B_bf16_tp2.nemo \
                            prompt_learning.model.tensor_model_parallel_size=2 \
                            >> /results/prompt_learning_gpt3_log.txt 2>&1"
      
      #job data
      data = {
                "name": "p_tuning_train_gpt5b_airflow",
                "aceInstance": "dgxa100.80g.4.norm",
                "aceName": ace,
                "dockerImageName": f"{org}/nemofw-training:23.05-py3",
                "jobOrder": 50,
                "jobPriority": "NORMAL",
                "replicaCount": 1,
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
                        "containerMountPoint": "/mount/workspace",
                        "id": workspace_id,
                        "mountMode": "RW"
                    }
                ],
                "command": p_tuning_command
            } 

      job_response = ngc_job_request(ti, org, data, team)
      job_id = job_response['job']['id']

      #keep waiting until job completes on bcp before ending airflow task
      job_status = ngc_job_status(ti, org, job_id)
      while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED':
            time.sleep(20)
            job_status = ngc_job_status(ti, org, job_id)
            print(job_status)

      return job_response 
    
# def p_tuning_bcp_inference(ti, org, ace):
#       #get workspace id
#       workspace_response = ti.xcom_pull(task_ids='workspace')
#       workspace_id = workspace_response['workspace']['id']

#       #put together inference cmd - uses example inference p-tuning script already available in NeMo
#       inference_cmd = "cd ../; python3 opt/NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning_eval.py \
#                                 --config-path=/opt/NeMo/examples/nlp/language_modeling/conf/ \
#                                 --config-name=megatron_gpt_prompt_learning_inference.yaml \
#                                 gpt_model_file=/mount/workspace/gpt_models/nemo_gpt5B_bf16_tp2.nemo\
#                                 virtual_prompt_model_file=/mount/p-tuning-gpt/p_tuned_models/p_tuning_5b.nemo \ //FIX!!!!!!
#                                 data_paths=['/mount/data/SQuAD/squad_test.jsonl'] \ //FIX!!!!!!!
#                                 pred_file_path=/mount/p-tuning-gpt/inference_results/5b_inference_results.txt \ //FIX!!!!!!!
#                                 trainer.devices=2 \
#                                 tensor_model_parallel_size=2 \
#                                 pipeline_model_parallel_size=1"
      

#       #job data
#       data = {
#                 "name": "p_tuning_train_gpt5b_airflow",
#                 "aceInstance": "dgxa100.80g.4.norm",
#                 "aceName": ace,
#                 "dockerImageName": f"{org}/nemofw-training:23.05-py3",
#                 "jobOrder": 50,
#                 "jobPriority": "NORMAL",
#                 "replicaCount": 1,
#                 "reservedLabels": [],
#                 "resultContainerMountPoint": "/results",
#                 "runPolicy": {
#                     "preemptClass": "RUNONCE"
#                 },
#                 "systemLabels": [],
#                 "userLabels": [],
#                 "userSecretsSpec": [],
#                 "workspaceMounts": [
#                     {
#                         "containerMountPoint": "/mount/workspace",
#                         "id": workspace_id,
#                         "mountMode": "RW"
#                     }
#                 ],
#                 "command": inference_cmd_command
#             } 

#       job_response_json = ngc_job_request(ti, org, data)
#       return job_response_json 

## Define DAG + Tasks
with DAG(
         "P_TUNING_NEMO_BCP", 
         schedule_interval='@daily',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag:

    t1 = PythonOperator(
            task_id = 'token',
            python_callable=get_token,
            op_kwargs={"org": org_ , "team": team_},
            dag = dag
    ) 

    t2 = PythonOperator(
            task_id = 'workspace',
            python_callable= create_workspace,
            op_kwargs= {"org":org_, "ace": ace_, "workspace_name": workspace_name_},
            dag = dag
    )

    t3 = PythonOperator(
            task_id = 'download_nemo_checkpoint',
            python_callable= download_nemo_checkpoint,
            op_kwargs= {"org":org_, "ace": ace_, "team": team_},
            dag = dag
          
    )

    t4 = PythonOperator(
            task_id = 'p_tuning_train',
            python_callable= p_tuning_training_bcp,
            op_kwargs= {"org":org_, "ace": ace_, "team": team_},
            dag = dag
    )

t1 >> t2 >> t3 >> t4