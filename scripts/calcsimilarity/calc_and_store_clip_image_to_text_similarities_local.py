import pickle
import sys
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

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
    parser.add_argument('--dataframe_path', type=str, default=os.path.join('ilsvrc2012', 'imagenet_captions.parquet'))
    parser.add_argument('--index2filename_path', type=str, default=None)

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
    image_to_text_sim_col = 'image_to_text_similarity'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    df = pd.read_parquet(params['dataframe_path'])

    # Create a new column
    if image_to_text_sim_col not in df:
        df[image_to_text_sim_col] = np.nan

    # Find rows w/o similarity
    df_todo = df.iloc[np.isnan(df[image_to_text_sim_col].tolist())]

    print_verbose('done!\n')

    # ----- Load the map to files -----
    print_verbose('loading index2filename map ...')

    if params['index2filename_path'] is None:
        print_verbose('\tdefaulting to identical map ...')
        index2filename = {idx: idx for idx in df.index}
    else:
        with open(params['index2filename_path'], 'rb') as f:
            index2filename = pickle.load(f)

    print_verbose('done!\n')

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    indices = []
    similarities = []
    errors = []

    indices_batch = []
    texts_batch = []
    images_batch = []
    i_batch = 0
    i_row = -1
    for idx, row in tqdm(df_todo.iterrows(), desc='calc. image-text sim.', total=len(df_todo)):
        i_row += 1

        # Parse
        file_name = index2filename[idx]
        text = row[configs.LAIONConfig.TEXT_COL]

        # Load the image
        file_path = os.path.join(params['images_path'], file_name)
        image = Image.open(file_path)

        if image.mode == 'RGB':
            # Add to the batch
            indices_batch.append(idx)
            texts_batch.append(text)
            images_batch.append(image)

        if len(indices_batch) < configs.CLIPConfig.BATCH_SIZE and i_row < (len(df_todo) - 1):
            continue
        if len(indices_batch) == 0:
            continue

        # Calc. embeddings
        similarities_batch, errors_batch = \
            calc_image_to_text_similarities(images_batch, texts_batch, clip)

        for error in errors_batch:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Update
        indices.extend(indices_batch)
        similarities.extend(similarities_batch)

        # Save
        if (i_batch + 1) % params['save_freq'] == 0:
            print_verbose('saving ....')
            df.loc[indices, image_to_text_sim_col] = similarities
            df.to_parquet(params['dataframe_path'], index=True)

            indices = []
            similarities = []

            print_verbose('done!\n')

        # Step
        indices_batch = []
        texts_batch = []
        images_batch = []
        i_batch += 1

    # ----- Save ------
    print_verbose('saving error logs ....')

    err_file_path = params['dataframe_path'].replace('.parquet', '_imgtxtsim_errors.txt')
    with open(err_file_path, 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')

    print_verbose('saving updated dataframe ....')

    if len(indices) > 0:
        df.loc[indices, image_to_text_sim_col] = similarities
        df.to_parquet(params['dataframe_path'], index=True)
    else:
        print_verbose('\talready saved!')

    print_verbose('done!\n')
