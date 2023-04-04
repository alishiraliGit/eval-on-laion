import os
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import configs

drive_service = None


def authenticate(cred_path):
    flow = InstalledAppFlow.from_client_secrets_file(cred_path, configs.GDriveConfig.SCOPES)
    creds = flow.run_local_server(port=0)

    return creds


def build_service(creds):
    global drive_service
    drive_service = build('drive', 'v3', credentials=creds)


def get_file_ids(folder_id):
    global drive_service

    # Set the query for files in the folder with the specified folder_id
    query = f"'{folder_id}' in parents"

    # Set the fields to retrieve only the file ID and file name
    fields = "nextPageToken, files(id, name)"

    # Set the page size to retrieve all files in one request
    page_size = 1000

    # Initialize the results variable
    results = []

    # Retrieve the first page of files
    response = drive_service.files().list(q=query, fields=fields, pageSize=page_size).execute()
    results.extend(response.get('files', []))

    # Retrieve the remaining pages of files
    while response.get('nextPageToken'):
        page_token = response.get('nextPageToken', None)
        response = drive_service.files().list(q=query, fields=fields, pageSize=page_size, pageToken=page_token).execute()
        results.extend(response.get('files', []))

    # Print the file IDs and names of all files in the folder
    fileid2name = {}
    for file in results:
        fileid2name[file['id']] = file['name']

    return fileid2name


def download_file(file_id, file_name, save_path):
    global drive_service

    try:
        # Download the file
        request = drive_service.files().get_media(fileId=file_id)
        file = io.BytesIO(request.execute())

        # Save the file to your local system
        with open(os.path.join(save_path, file_name), 'wb') as f:
            f.write(file.getbuffer())

        print(f'{file_name} downloaded successfully')
    except HttpError as error:
        print(f'An error occurred: {error}')
        file = None

    return file
