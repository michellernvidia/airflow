# import os, json, base64, requests
# from datetime import datetime
# from airflow import DAG
# from airflow.decorators import task
# from airflow.operators.python import PythonOperator
# from airflow.models import Variable

# ## 0. Variables
# key_v = Variable.get("key_v", deserialize_json=True)
# org_v = Variable.get("org_v", deserialize_json=True)
# team_v = Variable.get("team_v", deserialize_json=True)
# ace_v = Variable.get("ace_v", deserialize_json=True)
# nemo_ckpt_v = Variable.get("nemo_ckpt_v", deserialize_json=True)

# key_= str(key_v)
# org_=str(org_v)
# team_= str(team_v)
# ace_=str(ace_v)
# nemo_ckpt_=str(nemo_ckpt_v)

# ## 1. Connect to BCP API
# def get_token(ti, org):
#         api = ti.xcom_pull(task_ids='api_connect')
#         print(f"Xcom pull gives me {api}")
#         print(f"idk if this will work but here's ti {ti}")
        
#         '''Use the api key set environment variable to generate auth token'''
#         scope = f'group/ngc:{org}'
#         # if team: #shortens the token if included
#         #   scope += f'/{team}'
#         querystring = {"service": "ngc", "scope": scope}
#         auth = '$oauthtoken:{0}'.format(api)
#         auth = base64.b64encode(auth.encode('utf-8')).decode('utf-8')
        
#         headers = {
#             'Authorization': f'Basic {auth}',
#             'Content-Type': 'application/json',
#             'Cache-Control': 'no-cache',
#          }
#         url = 'https://authn.nvidia.com/token'
#         response = requests.request("GET", url, headers=headers, params=querystring)
#         if response.status_code != 200:
#              print(response)
#              raise Exception("HTTP Error %d: from %s" % (response.status_code, url))
#         return json.loads(response.text.encode('utf8'))["token"]

# ## 2. Download a NeMo checkpoint into a Workspace 

# def create_workspace(ti, org, ace, name):
#         token = ti.xcom_pull(task_ids='token')
#         print(f"Xcom pull gives me {token}")
        
#         '''Create a workspace in a given org for the authenticated user'''
#         url = f'https://api.ngc.nvidia.com/v2/org/{org}/workspaces/'
#         headers = {
#             'Content-Type': 'application/json',
#             'Authorization': f'Bearer {token}'
#          }
#         data = {
#           'aceName': f'{ace}',
#           'name': nemo_ckpt_.split('.')[0] + '_workspace'
#          }
#         response = requests.request("POST", url, headers=headers, data=json.dumps(data))
#         if response.status_code != 200:
#             raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
        
#         return response.json()

# def download_nemo_checkpoint(ti):
#       '''Downloads pretrained GPT .nemo checkpoint into our created bcp workspace'''
#       workspace_id = ti.xcom_pull(task_ids='api_connect')
#       url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/'

#       headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}

#       ## currently configured for 5b 
#       data = {"aceInstance": "dgxa100.80g.1.norm",
#             "aceName": "launchpad-iad2-ace",
#             "command": "cd ../; cd /mount/data/; \
#                 wget https://huggingface.co/nvidia/nemo-megatron-gpt-5B/resolve/main/nemo_gpt5B_bf16_tp2.nemo",
#             "dockerImageName": "tzcwjedpb1di/nemofw-training:23.04.1-py3",
#             "jobOrder": 50,
#             "jobPriority": "NORMAL",
#             "name": "airflow_download_gpt3_5b_ckpt",
#             "replicaCount": 1,
#             "reservedLabels": [],
#             "resultContainerMountPoint": "/results",
#             "runPolicy": {
#                 "preemptClass": "RUNONCE"
#             },
#             "systemLabels": [],
#             "userLabels": [],
#             "userSecretsSpec": [],
#             "workspaceMounts": [
#                 {
#                     "containerMountPoint": "/mount/data",
#                     "id": workspace_id,
#                     "mountMode": "RW"
#                 }
#             ]
#             }

      

# ## 3. Launch P-tuning on BCP to tune the saved NeMo Checkpoint


# ## Define DAG + Tasks