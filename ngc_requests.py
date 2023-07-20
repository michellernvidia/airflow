import json, base64, requests, time


##api key. don't steal pls. 
KEY = "bGxzOGN0NThja25jbGhhMWduaW5ya3ZuMTM6YjcxMWE5ZWItMjdmNC00MTA5LWI5NmItMTkzNDJiMzg0ZjFk"


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


# NEEDS TO BE RE-RUN + TESTED ON AIRFLOW
def ngc_job_request(ti, org, job_name, ace_instance, ace_name, docker_image, replica_count, \
                    workspace_mount_path, workspace_id, job_command, team=None, \
                    multinode=False, array_type=None, total_runtime=None):
      '''Creates an NGC job request via API'''
      token = ti.xcom_pull(task_ids='token')
      if team:
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/jobs/'
      else:
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/'

      headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}

      if multinode:
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
                    "command": job_command,
                    "arrayType": array_type,
                    "totalRuntime": total_runtime
                }
          
      else:
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
    
    # token = ti.xcom_pull(task_ids='token')
    #because we will have to get the ngc job status continously, we have to 
    #get a new token each time to make sure the token doesn't expire
    token = get_token(ti, key=KEY, org=org, team=None)

    print(f'TOKEN: {token}')
    url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/{job_id}'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.request("GET", url, headers=headers)
    if response.status_code != 200:
        print('JOB STATUS RESPONSE is not 200. This is wot we got: ', response)
        raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
    job_info = response.json()
    return job_info['job']['jobStatus']['status']


def wait_for_job_completion(ti, org, job_response, wait_time, team=None):
    '''Continually gets NGC job status every `wait_time` seconds until 
    the job finishes, is killed, or fails'''
    
    job_id = job_response['job']['id']
    job_status = ngc_job_status(ti, org, job_id)
    min=0
    while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED' and job_status != 'KILLED_BY_USER':
        time.sleep(wait_time)
        min+=5
        job_status=ngc_job_status(ti, org, job_id)
        print(f'minute: {min} | Job status: ', job_status)
    
    return job_status

    # job_id = job_response['job']['id']
    # job_status = ngc_job_status(ti, org, job_id)
    # while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED' and job_status != 'KILLED_BY_USER':
    #         time.sleep(20)
    #         job_status = ngc_job_status(ti, org, job_id)
    #         print(job_status)
