import os
from google.cloud import storage

import io 
import torch


def save_model_state_dict_to_gcs(bucket_name, destination_blob_name, model):
    
    if not destination_blob_name.endswith(".pth"):
        destination_blob_name += ".pth" 

    #
    # Create a buffer to hold the state_dict
    #
    buffer = io.BytesIO()
    
    #
    # Save only the state_dict, which is a dictionary of model weights
    #
    torch.save(model.state_dict(), buffer)
    
    #
    # Seek to the beginning of the buffer to prepare for uploading
    #
    buffer.seek(0)

    #
    # Initialize a GCS client
    #
    storage_client = storage.Client()

    #
    # Get the GCS bucket
    #
    bucket = storage_client.bucket(bucket_name)

    #
    # Create a blob object with the desired destination name
    #
    blob = bucket.blob(destination_blob_name)

    #
    # Upload the state_dict as a file
    #
    blob.upload_from_file(buffer, content_type="application/octet-stream")

    print(f"Model state_dict uploaded to {destination_blob_name}.")


def load_model_state_dict_from_gcs(bucket_name, blob_name, model):

    if not blob_name.endswith(".pth"):
        blob_name += ".pth"

    #
    # Initialize a GCS client
    #
    storage_client = storage.Client()
    
    #
    # Get the GCS bucket and blob
    #
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    #
    # Create a buffer to hold the downloaded data
    #
    buffer = io.BytesIO()
    
    #
    # Download the blob content to the buffer
    #
    blob.download_to_file(buffer)
    
    #
    # Seek to the beginning of the buffer
    #
    buffer.seek(0)

    #
    # Load the state_dict into the model
    #
    model.load_state_dict(torch.load(buffer, map_location='cpu'))  # Load to CPU first
    
    return model

def load_model_state_dict_from_local(file_path, model, map_location=None):
    if not file_path.endswith(".pth"):
        file_path += ".pth"

    if not os.path.exists(file_path):
        return None

    #
    # Load the state_dict from the local file
    #
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load(file_path, map_location=map_location))
    
    return model