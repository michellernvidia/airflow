from ngc_requests import get_existing_workspace, create_workspace

def create_task_workspace(ti, ngc_api_key, org, ace, workspace_name):
    '''Creates a NGC workspace in the specied NGC org and ace under the name *workspace_name*.
    Returns workspace ID for created workspace to be used in downstream tasks.'''
    
    #check if workspace exists already
    workspace_response=get_existing_workspace(ti, ngc_api_key, org, workspace_name)
    
    #if not, create a new one
    if workspace_response["requestStatus"]["statusCode"] == 'NOT_FOUND':
        print(f'Workspace does not exist. Creating workspace {workspace_name}.')
        workspace_response = create_workspace(ti, ngc_api_key, org, ace, workspace_name)
    else:
        print(f'Workspace {workspace_name} exists already.')
    
    workspace_id = workspace_response['workspace']['id']
    return workspace_id