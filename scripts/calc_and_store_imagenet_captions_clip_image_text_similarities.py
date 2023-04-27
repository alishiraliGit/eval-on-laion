import sys
import os
import json
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from core.clip import CLIP


def calc_image_to_text_similarities(imgs, txts, clip_mdl: CLIP):
    # Calc. similarities
    errs = []
    try:
        sims = clip_mdl.similarities(texts=txts, images=imgs)
        return sims, errs
    except Exception as e:
        errs.append({'cause': 'In calc. image-text similarities an error occurred.', 'error': e})
        return [np.nan]*len(imgs), errs


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_train_selected'))
    parser.add_argument('--imagenet_captions_path', type=str,
                        default=os.path.join('ilsvrc2012', 'imagenet_captions.json'))
    parser.add_argument('--save_path', type=str, default=os.path.join('ilsvrc2012'))

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--save_freq', type=int, default=1000)

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Compute
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Naming
    prefix = 'imagenet_captions'
    df_file_name = prefix + '.parquet'
    image_to_text_sim_col = 'image_to_text_similarity'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load imagenet-captions -----
    print_verbose('loading imagenet-captions ...')

    with open(params['imagenet_captions_path'], 'rb') as f:
        imagenet_captions = json.load(f)

    print_verbose('done!\n')

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    file_names = []
    texts = []
    similarities = []
    errors = []

    file_names_batch = []
    texts_batch = []
    images_batch = []
    i_batch = 0
    for i_ic, ic in tqdm(enumerate(imagenet_captions), desc='calc. image-text sim.', total=len(imagenet_captions)):
        # Make a caption
        text = ic['title'] + ' ' + ic['description']

        # Load the image
        file_name = ic['filename']
        file_path = os.path.join(params['images_path'], file_name)
        image = Image.open(file_path)

        if image.mode != 'RGB':
            continue

        # Add to the batch
        file_names_batch.append(file_name)
        texts_batch.append(text)
        images_batch.append(image)

        if len(images_batch) < configs.CLIPConfig.BATCH_SIZE and i_ic < len(imagenet_captions):
            continue

        # Calc. embeddings
        similarities_batch, errors_batch = \
            calc_image_to_text_similarities(images_batch, texts_batch, clip)

        for error in errors_batch:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Update
        file_names.extend(file_names_batch)
        texts.extend(texts_batch)
        similarities.extend(similarities_batch)

        # Save
        if (i_batch + 1) % params['save_freq'] == 0:
            print_verbose('saving ....')

            df = pd.DataFrame(
                {
                    configs.LAIONConfig.TEXT_COL: texts,
                    image_to_text_sim_col: similarities
                },
                index=file_names
            )

            df.to_parquet(os.path.join(params['save_path'], df_file_name), index=True)

            print_verbose('done!\n')

        # Empty current batch
        file_names_batch = []
        texts_batch = []
        images_batch = []
        i_batch += 1

    # ----- Save ------
    print_verbose('saving error logs ....')

    err_file_name = df_file_name.replace('.parquet', '_errors.txt')
    with open(os.path.join(params['save_path'], err_file_name), 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')

    print_verbose('saving ....')

    df = pd.DataFrame(
        {
            configs.LAIONConfig.TEXT_COL: texts,
            image_to_text_sim_col: similarities
        },
        index=file_names
    )

    df.to_parquet(os.path.join(params['save_path'], df_file_name), index=True)

    print_verbose('done!\n')
