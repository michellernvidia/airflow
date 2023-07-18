import json, base64, requests, time

KEY='bGxzOGN0NThja25jbGhhMWduaW5ya3ZuMTM6YjcxMWE5ZWItMjdmNC00MTA5LWI5NmItMTkzNDJiMzg0ZjFk'
ORG="iffx7vlsrd5t"
TEAM="nvbc-interns"
ACE="nv-launchpad-bc-iad1-ace4"

def get_token(key, org=None, team=None):
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


def ngc_job_status(org, job_id):
    '''Gets status of NGC Job (e.g., SUCCESS, FAILED, CREATED, etc.)'''
    
    token = get_token(KEY, org, team=None)
    url = f'https://api.ngc.nvidia.com/v2/org/{org}/jobs/{job_id}'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.request("GET", url, headers=headers)
    if response.status_code != 200:
        print('JOB STATUS RESPONSE', response, response.json())
        raise Exception("HTTP Error %d: from '%s'" % (response.status_code, url))
    job_info = response.json()
    return job_info['job']['jobStatus']['status']

def main():
    job_id = 4887671 #4888173
    job_status = ngc_job_status(ORG, job_id)
    min=0
    print(f'MINUTE:{min}',job_id, job_status)
    while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED' and job_status != 'KILLED_BY_USER':
        time.sleep(300) #increase wait time to 5 mins - 401 error somewhere between 5-10 mins
        min+=5
        job_status = ngc_job_status(ORG, job_id)
        print(f'MINUTE:{min}',job_id, job_status)


if __name__ == "__main__":
    main()