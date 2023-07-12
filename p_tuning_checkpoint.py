import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable

## 0. Variables
key_v = Variable.get("key_v", deserialize_json=True)
org_v = Variable.get("org_v", deserialize_json=True)
team_v = Variable.get("team_v", deserialize_json=True)
ace_v = Variable.get("ace_v", deserialize_json=True)
workspace_name_v = Variable.get("workspace_name_v", deserialize_json=True)
nemo_ckpt_v = Variable.get("nemo_ckpt_v", deserialize_json=True)
pretrain_decision_v = Variable.get("pretrain_decision_v", deserialize_json=True)

key_= str(key_v)
org_=str(org_v)
team_= str(team_v)
ace_=str(ace_v)
workspace_name_ = str(workspace_name_v)
nemo_ckpt_=str(nemo_ckpt_v)
pretrain_decision_ = str(pretrain_decision_v)


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


def choose_pretrain_path(ti):
    if pretrain_decision_ == False:
        return 'download_nemo_checkpoint'
    else:
        return 'pretrain_gpt_model'
    

def download_nemo_checkpoint(ti, org, ace, team=None):
      
      #get workspace id
      workspace_response = ti.xcom_pull(task_ids='workspace')
      workspace_id = workspace_response['workspace']['id']
      
      #ngc job parameters
      job_name = "airflow_download_gpt3_5b_ckpt"
      ace_instance = "dgxa100.80g.1.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.05-py3"
      replica_count = 1
      workspace_mount_path = "/mount/data"
      job_command = "cd ../; cd /mount/data/; mkdir gpt_models; cd gpt_models;\
                    wget https://huggingface.co/nvidia/nemo-megatron-gpt-5B/resolve/main/nemo_gpt5B_bf16_tp2.nemo"
      
      #send ngc job request
      job_response = ngc_job_request(ti, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspace_mount_path, workspace_id, job_command, team=team)
      
      #wait for job to complete on BCP before allowing airflow to "finish" task
      job_id = job_response['job']['id']
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
      
      #ngc job parameters
      job_name = "p_tuning_train_gpt5b_airflow"
      ace_instance = "dgxa100.80g.4.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.05-py3"
      replica_count = 1
      workspace_mount_path = "/mount/workspace"
      job_command = "python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
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
      
      #send ngc job request
      job_response = ngc_job_request(ti, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspace_mount_path, workspace_id, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      job_id = job_response['job']['id']
      job_status = ngc_job_status(ti, org, job_id)
      while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED':
            time.sleep(300) #increase wait time to 5 mins
            job_status = ngc_job_status(ti, org, job_id)
            print(job_status)

      return job_response 
    

def download_pile_dataset(ti, org, ace, team=None):
     return

## Define DAG + Tasks
with DAG(
         "P_TUNING_NEMO_BCP", 
         schedule_interval='@daily',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag:

    token_task = PythonOperator(
            task_id = 'token',
            python_callable=get_token,
            op_kwargs={"org": org_ , "team": team_},
            dag = dag
    ) 

    workspace_task = PythonOperator(
            task_id = 'workspace',
            python_callable= create_workspace,
            op_kwargs= {"org":org_, "ace": ace_, "workspace_name": workspace_name_},
            dag = dag
    )

    pretrain_decision_task = BranchPythonOperator(
            task_id='decide_pretrain_LLM',
            provide_context=True,
            python_callable=choose_pretrain_path,
            dag=dag)
    
    download_checkpoint_task = PythonOperator(
            task_id = 'download_nemo_checkpoint',
            python_callable= download_nemo_checkpoint,
            op_kwargs= {"org":org_, "ace": ace_, "team": team_},
            dag = dag
          
    )

    download_the_pile_task = PythonOperator(
            task_id = 'download_pile_dataset',
            python_callable= download_pile_dataset,
            op_kwargs= {"org":org_, "ace": ace_, "team": team_},
            dag = dag
          
    )

    # p_tuning_train_task = PythonOperator(
    #         task_id = 'p_tuning_train',
    #         python_callable= p_tuning_training_bcp,
    #         op_kwargs= {"org":org_, "ace": ace_, "team": team_},
    #         dag = dag
    # )

# t1 >> t2 >> t3 >> t4
token_task >> workspace_task >> pretrain_decision_task >> [download_checkpoint_task, download_the_pile_task]