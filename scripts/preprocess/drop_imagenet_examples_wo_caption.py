import sys
import os
import json
import shutil
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Settings -----
    imagenet_captions_path = os.path.join('imagenet-captions', 'imagenet_captions.json')
    images_path = os.path.join('ilsvrc2012', 'ILSVRC2012_img_train')
    selected_images_path = os.path.join('ilsvrc2012', 'ILSVRC2012_img_train_selected')

    # ----- Init -----
    logu.verbose = True

    # ----- Load -----
    # Imagenet-captions
    print_verbose('loading imagenet-captions ...')

    with open(imagenet_captions_path, 'rb') as f:
        imagenet_captions = json.load(f)

    print_verbose('done!\n')

    # ----- Find images with caption -----
    ic_file_names = [ic['filename'] for ic in imagenet_captions]

    for file_name in tqdm(ic_file_names):
        shutil.copy2(os.path.join(images_path, file_name), os.path.join(selected_images_path, file_name))
