import sys
import os
import argparse
import pickle
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
from core.queries import select_queries, QueryType


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_train_selected'))
    parser.add_argument('--dataframe_path', type=str, default=os.path.join('ilsvrc2012', 'imagenet_captions.parquet'))
    parser.add_argument('--index2filename_path', type=str, default=None)

    parser.add_argument('--index2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'labels', 'icimagename2wnid.pkl'))

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)

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

    # Query
    query_func = select_queries([params['query_type']])[0]

    # Column names
    query_col = params['query_type'] + '_' + 'wnid'
    image_to_query_sim_col = f'image_to_{query_col}_similarity'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    df = pd.read_parquet(params['dataframe_path'])

    # Create a new column
    if image_to_query_sim_col not in df:
        df[image_to_query_sim_col] = np.nan

    # Find rows w/o similarity
    df_todo = df.iloc[np.isnan(df[image_to_query_sim_col].tolist())]

    print_verbose('done!\n')

    # ----- Load labels -----
    print_verbose('loading labels ...')

    with open(os.path.join(params['index2wnid_path']), 'rb') as f:
        index2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Design queries -----
    print_verbose('design queries ...')

    index2query = {}
    for idx, wnid in index2wnid.items():
        index2query[idx] = query_func(wnid)

    # Add to df
    df[query_col] = df.index.map(index2query)

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

    # ----- Load and calc. similarity -----
    all_indices = []
    all_similarities = []

    indices_batch = []
    images_batch = []
    i_batch = 0
    i_row = -1
    for idx, row in tqdm(df_todo.iterrows(), desc='calc. image-query sim.', total=len(df_todo)):
        i_row += 1

        # Load the image
        file_name = index2filename[idx]
        file_path = os.path.join(params['images_path'], file_name)
        image = Image.open(file_path)

        if image.mode == 'RGB':
            indices_batch.append(idx)
            images_batch.append(image)

        if len(indices_batch) < configs.CLIPConfig.BATCH_SIZE and i_row < (len(df_todo) - 1):
            continue
        if len(indices_batch) == 0:
            continue

        # Get the queries
        queries_batch = df.loc[indices_batch, query_col].tolist()

        # Find image-to-query similarities
        similarities_batch = clip.similarities(texts=queries_batch, images=images_batch)

        # Step
        all_indices.extend(indices_batch)
        all_similarities.extend(similarities_batch)

        images_batch = []
        indices_batch = []
        i_batch += 1

        # Save
        if (i_batch + 1) % params['save_freq'] == 0:
            print_verbose('saving ....')

            df.loc[all_indices, image_to_query_sim_col] = all_similarities
            df.to_parquet(params['dataframe_path'], index=True)

            all_indices = []
            all_similarities = []

            print_verbose('done!\n')

    # ----- Save error logs ------
    print_verbose('saving ....')

    if len(all_indices) > 0:
        df.loc[all_indices, image_to_query_sim_col] = all_similarities
        df.to_parquet(params['dataframe_path'], index=True)
    else:
        print_verbose('\talready saved!')

    print_verbose('done!\n')
