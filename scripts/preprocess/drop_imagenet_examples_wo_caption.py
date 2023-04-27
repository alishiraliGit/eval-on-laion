import sys
import os
import json
import glob
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Settings -----
    imagenet_captions_path = os.path.join('ilsvrc2012', 'imagenet_captions.json')
    images_path = os.path.join('ilsvrc2012', 'ILSVRC2012_img_train')

    # ----- Init -----
    logu.verbose = True

    # ----- Load -----
    # Imagenet-captions
    print_verbose('loading imagenet-captions ...')

    with open(imagenet_captions_path, 'rb') as f:
        imagenet_captions = json.load(f)

    print_verbose('done!\n')

    # ----- Find images with caption -----
    file_paths = glob.glob(os.path.join(images_path, '*.JPEG'))
    ic_file_paths = [ic['filename'] for ic in imagenet_captions]

    not_found_paths = []
    for i_f, file_path in tqdm(enumerate(file_paths)):
        if file_path not in ic_file_paths:
            not_found_paths.append(not_found_paths)

        if i_f % 1000 == 0:
            print_verbose(f'so far, could not find {len(not_found_paths)} of the images out of {i_f + 1}.')
