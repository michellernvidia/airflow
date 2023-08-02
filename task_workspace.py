from ngc_requests import *


def create_tuning_workspace(ti, ngc_api_key, org, ace, workspace_name):
    workspace_response = create_workspace(ti, ngc_api_key, org, ace, workspace_name)
    workspace_id = workspace_response['workspace']['id']
    return workspace_id

def create_gpt_workspace(ti, ngc_api_key, org, ace, workspace_name):
    workspace_response = create_workspace(ti, ngc_api_key, org, ace, workspace_name)
    workspace_id = workspace_response['workspace']['id']
    return workspace_id