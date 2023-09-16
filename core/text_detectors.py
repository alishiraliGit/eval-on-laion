import os
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import urllib.request

import configs
from utils import cv2_utils as cv2u
from utils.logging_utils import print_verbose
from utils.utils import extract_tar_gz


class EAST:
    def __init__(self, ver=configs.EASTConfig.DEFAULT_VERSION):
        self.ver = ver

        self.net = cv2.dnn.readNet(os.path.join(configs.EASTConfig.PATH, ver + configs.EASTConfig.DEFAULT_FILE_EXT))
        self._layerNames = [
            'feature_fusion/Conv_7/Sigmoid',
            'feature_fusion/concat_3'
        ]

    @staticmethod
    def download(ver=configs.EASTConfig.DEFAULT_VERSION):
        os.makedirs(configs.EASTConfig.PATH, exist_ok=True)
        east_net_path = os.path.join(configs.EASTConfig.PATH, ver + configs.EASTConfig.DEFAULT_FILE_EXT)
        if not os.path.exists(east_net_path):
            print_verbose(f'downloading {ver} to {east_net_path} ...')

            east_net_tar_path = os.path.join(configs.EASTConfig.PATH, ver + '.tar.gz')
            urllib.request.urlretrieve(
                url=configs.EASTConfig.URL,
                filename=east_net_tar_path
            )

            extract_tar_gz(east_net_tar_path, configs.EASTConfig.PATH)

            print_verbose('done!\n')

    @staticmethod
    def select_box(image_cv2, box):
        start_x, start_y, end_x, end_y = box

        return image_cv2[start_y: end_y][:, start_x: end_x]

    def detect_and_select_boxes(self, image_content, draw_box=True):
        image_annotated, boxes_cv2 = self.detect(image_content, draw_box=draw_box)

        image_cv2 = cv2u.to_cv2(image_content)
        image_boxes = []
        for box_cv2 in boxes_cv2:
            image_box = cv2u.from_cv2(self.select_box(image_cv2, box_cv2))
            image_boxes.append(image_box)

        return image_annotated, image_boxes

    def detect(self, image_content, draw_box=True):
        image_cv2 = cv2u.to_cv2(image_content)

        orig = image_cv2.copy()

        if len(image_cv2.shape) == 2:
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_GRAY2RGB)

        (h, w) = image_cv2.shape[:2]

        # Set the new width and height and then determine the ratio in change
        # for both the width and height: Should be multiple of 32
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32

        r_w = w / float(new_w)
        r_h = h / float(new_h)

        # Resize the image and grab the new image dimensions
        image_cv2 = cv2.resize(image_cv2, (new_w, new_h))

        (h, w) = image_cv2.shape[:2]

        # Forward the image
        blob = cv2.dnn.blobFromImage(image_cv2, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        self.net.setInput(blob)

        (scores, geometry) = self.net.forward(self._layerNames)

        # Find the boxes
        (num_rows, num_cols) = scores.shape[2:4]
        rects = []
        confidences = []
        # Loop over the number of rows
        for y in range(0, num_rows):
            # Extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that surround text
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(0, num_cols):
                # If our score does not have sufficient probability, ignore it
                # Set minimum confidence as required
                if scores_data[x] < 0.5:
                    continue
                # Compute the offset factor as our resulting feature maps will x smaller than the input image
                (offset_x, offset_y) = (x * 4.0, y * 4.0)
                # Extract the rotation angle for the prediction and then compute the sin and cosine
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # Use the geometry volume to derive the width and height of the bounding box
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                # Compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
                end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                # Add the bounding box coordinates and probability score to our respective lists
                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # Loop over the bounding boxes
        scaled_boxes = []
        for (start_x, start_y, end_x, end_y) in boxes:
            # Scale the bounding box coordinates based on the respective ratios
            start_x = int(start_x * r_w)
            start_y = int(start_y * r_h)
            end_x = int(end_x * r_w)
            end_y = int(end_y * r_h)

            scaled_boxes.append((start_x, start_y, end_x, end_y))

            # Draw the bounding box on the image
            if draw_box:
                cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        return cv2u.from_cv2(orig), scaled_boxes[:configs.EASTConfig.MAX_NUM_BOXES]
