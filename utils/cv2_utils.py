from PIL import Image
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression


def to_cv2(image_content):
    # Convert the content into a NumPy array
    image_array = np.frombuffer(image_content, np.uint8)

    # Decode the NumPy array into an OpenCV image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


def from_cv2(image_cv2):
    # Convert the OpenCV image to a NumPy array
    image_np = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create a PIL image from the NumPy array
    image_pil = Image.fromarray(image_np)

    return image_pil
