import pickle
import sys
import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from core.clip import CLIP
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))

    parser.add_argument('--save_path', type=str, default=os.path.join('ilsvrc2012', 'processed'))

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Find the data -----
    image_file_paths = [
        os.path.join(params['images_path'], 'ILSVRC2012_val_%08d.JPEG' % (idx + 1))
        for idx in range(configs.ILSVRCConfigs.NUM_VAL)
    ]
    n_image = len(image_file_paths)

    # ----- Loop over batches -----
    embeds = np.zeros((n_image, configs.CLIPConfig.DIM))

    for idx in tqdm(range(0, n_image, configs.CLIPConfig.BATCH_SIZE)):
        idx_to = np.minimum(idx + configs.CLIPConfig.BATCH_SIZE, n_image)

        # Load the batch
        paths = image_file_paths[idx: idx_to]
        images = []
        for path in paths:
            image = Image.open(path)
            images.append(image)

        # Calc. the embeddings
        embeds_batch = clip.image_embeds(images)
        embeds[idx: idx_to] = embeds_batch

    # ----- Save -----
    print_verbose('saving embeddings ...')

    with open(os.path.join(params['save_path'], 'val_img_embeddings.pkl'), open_type) as f:
        pickle.dump(embeds, f)

    print_verbose('done!\n')
