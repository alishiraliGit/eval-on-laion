import sys
import os
import argparse
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils.ilsvrc_utils import load_lemmas_and_wnids
from core.clip import CLIP
from core.queries import select_queries, QueryType


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))
    parser.add_argument('--dataframe_path', type=str,
                        default=os.path.join('ilsvrc2012', 'ILSVRC2012_val_with_sims_to_all_queries.parquet'))
    parser.add_argument('--index2filename_path', type=str, default=None)

    parser.add_argument('--synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))

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
    image_to_query_col_func = lambda w: f'image_to_{params["query_type"]}_{w}_similarity'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load synsets -----
    print_verbose('loading synsets ...')

    id_lemmas_df = load_lemmas_and_wnids(params['synsets_path'])
    all_wnids = id_lemmas_df[configs.ILSVRCConfigs.WNID_COL].tolist()

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    df = pd.read_parquet(params['dataframe_path'])

    # Create new columns
    for wnid in all_wnids:
        if image_to_query_col_func(wnid) not in df:
            df[image_to_query_col_func(wnid)] = np.nan

    # Find rows w/o similarity
    df_todo = df.iloc[np.isnan(df[image_to_query_col_func(all_wnids[-1])].tolist())]

    print_verbose('done!\n')

    # ----- Find the embeddings for queries -----
    print_verbose('calc. embeddings for the queries')

    queries = [query_func(wnid) for wnid in all_wnids]
    q_embeds = clip.text_embeds(queries)
    q_embeds_norm = normalize(q_embeds, axis=1, norm='l2')

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
    for idx, row in tqdm(df_todo.iterrows(), desc='calc. image to all queries sim.', total=len(df_todo)):
        i_row += 1

        # Load the image
        file_path = os.path.join(params['images_path'], index2filename[idx])
        image = Image.open(file_path)

        if image.mode == 'RGB':
            # Add to the batch
            indices_batch.append(idx)
            images_batch.append(image)

        if len(indices_batch) < configs.CLIPConfig.BATCH_SIZE and i_row < (len(df_todo) - 1):
            continue
        if len(indices_batch) == 0:
            continue

        # Find image-to-query similarities
        try:
            # Get embeddings
            image_embeds_batch = clip.image_embeds(images_batch)
            image_embeds_batch_norm = normalize(image_embeds_batch, axis=1, norm='l2')

            # Find similarities
            similarity_to_queries_batch = image_embeds_batch_norm.dot(q_embeds_norm.T)

            # Step
            all_indices.extend(indices_batch)
            all_similarities.extend(similarity_to_queries_batch)

        except Exception as e:
            print(str(e))

        indices_batch = []
        images_batch = []
        i_batch += 1

        # Save
        if (i_batch + 1) % params['save_freq'] == 0:
            print_verbose('saving ....')

            df.loc[all_indices, [image_to_query_col_func(wnid) for wnid in all_wnids]] = np.array(all_similarities)
            df.to_parquet(params['dataframe_path'], index=True)

            all_indices = []
            all_similarities = []

            print_verbose('done!\n')

    # ----- Save error logs ------
    print_verbose('saving ....')

    if len(all_indices) > 0:
        df.loc[all_indices, [image_to_query_col_func(wnid) for wnid in all_wnids]] = np.array(all_similarities)
        df.to_parquet(params['dataframe_path'], index=True)
    else:
        print_verbose('\talready saved!')

    print_verbose('done!\n')
