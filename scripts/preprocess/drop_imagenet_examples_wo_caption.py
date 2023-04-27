import sys
import os
import json
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

    # Sanity check
    images_found = []
    for i_ic, ic in tqdm(enumerate(imagenet_captions)):
        if os.path.exists(os.path.join(images_path, ic['filename'])):
            images_found.append(i_ic)

        if i_ic % 1000 == 0:
            print_verbose(f'found {len(images_found)} so far.')

    print_verbose(f'found {len(images_found)} of imagenet-captions among the images.')
