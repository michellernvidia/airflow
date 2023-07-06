import os, json, base64, requests
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.models import Variable

org_v = Variable.get("org_v", deserialize_json=True)
ace_v = Variable.get("ace_v", deserialize_json=True)
name_v= Variable.get("name_v", deserialize_json=True)

org_=str(org_v)
ace_=str(ace_v)
name_=str(name_v)

def find_api_key(ti):
        expanded_conf_file_path = os.path.expanduser("~/.ngc/config")
        if os.path.exists(expanded_conf_file_path):
            print("Config file exists, pulling API key from it")
            try:
                config_file = open(expanded_conf_file_path, "r")
                lines = config_file.readlines()
                for line in lines:
                 if "apikey" in line:
                    elements = line.split()
                    return elements[-1]
                   
            except:
                print("Failed to find the API key in config file")
                return ''
        elif os.environ.get('API_KEY'):
            print("Using API_KEY environment variable")
            return os.environ.get('API_KEY')
            
        else:
            print("Could not find a valid API key")
            return ''
       
def get_token(ti, org ):
        api = ti.xcom_pull(task_ids='api_connect')
        print(f"Xcom pull gives me {api}")
        print(f"idk if this will work but here's ti {ti}")
        
        '''Use the api key set environment variable to generate auth token'''
        scope = f'group/ngc:{org}'
        # if team: #shortens the token if included
        #   scope += f'/{team}'
        querystring = {"service": "ngc", "scope": scope}
        auth = '$oauthtoken:{0}'.format(api)
        auth = base64.b64encode(auth.encode('utf-8')).decode('utf-8')
        
        headers = {
            'Authorization': f'Basic {auth}',
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
         }
        url = 'https://authn.nvidia.com/token'
        response = requests.request("GET", url, headers=headers, params=querystring)
        if response.status_code != 200:
             print(response)
             raise Exception("HTTP Error %d: from %s" % (response.status_code, url))
        return json.loads(response.text.encode('utf8'))["token"]

def create_workspace(ti, org, ace, name):
        token = ti.xcom_pull(task_ids='token')
        print(f"Xcom pull gives me {token}")
        
        '''Create a workspace in a given org for the authenticated user'''
        url = f'https://api.ngc.nvidia.com/v2/org/{org}/workspaces/'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
         }
        data = {
          'aceName': f'{ace}',
          'name': f'{name}'
         }
        response = requests.request("POST", url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
        return response.json()
        
with DAG(
         "API_workspace", 
         schedule_interval='@daily',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag:
    t1 = PythonOperator(
            task_id = 'api_connect',
            python_callable= find_api_key,
            dag = dag,          
    )
    t2 = PythonOperator(
            task_id = 'token',
            python_callable=get_token,
            op_kwargs={"org": org_},
            dag = dag
    )  
    t3 = PythonOperator(
            task_id = 'workspace',
            python_callable= create_workspace,
            op_kwargs= {"org":org_, "ace": ace_ , "name": name_},
            dag = dag
    )

t1 >> t2 >> t3