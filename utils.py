import os

def create_upload_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created upload directory: {directory}")
