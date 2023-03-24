import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO


import configs


def download_image(url):
    image = Image.open(BytesIO(requests.get(url, timeout=configs.RetrieveConfig.IMAGE_DOWNLOAD_TIMEOUT).content))

    if image.mode != 'RGB':
        raise Exception('Image mode is %s.' % image.mode)

    if np.prod(image.size) < configs.RetrieveConfig.MIN_IMAGE_SIZE:
        raise Exception('Image is too small.')

    return image


def download_and_save_image(url, save_path, file_name=None):
    # Send a GET request to the URL
    response = requests.get(url, timeout=configs.RetrieveConfig.IMAGE_DOWNLOAD_TIMEOUT)

    # Get the content type of the response
    content_type = response.headers['Content-Type']

    # Get the file extension from the content type
    if content_type == 'image/jpeg':
        extension = 'jpg'
    elif content_type == 'image/png':
        extension = 'png'
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

    # Get the file name from the URL
    if file_name is None:
        file_name = os.path.basename(url)

    # Construct the full file path with extension
    file_path = os.path.join(save_path, file_name + '.' + extension)

    # Write the response content to the file
    with open(file_path, 'wb') as f:
        f.write(response.content)
