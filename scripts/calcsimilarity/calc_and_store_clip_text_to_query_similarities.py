import sys
import os
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from core.clip import CLIP
from core.queries import select_queries, QueryType
from utils import laion_utils as laionu
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_filter', type=str, default='*')

    parser.add_argument('--lemma2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed',
                                             'lemma2wnid(unique_in_ilsvrc_ignored_empty_wnids).pkl'))

    # Method
    parser.add_argument('--method', type=str, help='Look at configs.LAIONConfig.')

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)
    parser.add_argument('--query_key', type=str, help='wnid or lemma')

    # Compute
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_gpu', action='store_true')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])
    logu.verbose = params['verbose']

    # Prefix
    prefix = configs.LAIONConfig.method_to_prefix(params['method'])

    # Query
    query_func = select_queries([params['query_type']])[0]

    # Column names
    query_col = params['query_type'] + '_' + params['query_key']
    sim_col = f'text_to_{query_col}_similarity'

    print_verbose('done!\n')

    # ----- Load the subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose(f'\tfound {len(df)} rows.')
    print_verbose('done!\n')

    # ----- Load maps and construct an inverse map -----
    print_verbose('loading labels (maps) and constructing an inverse map ...')

    maps_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'\tfound {len(maps_paths)} key2laion maps:\n')
    print_verbose('\t- ' + '\n\t- '.join(maps_paths))

    laionindex2keys = {}
    for path in tqdm(maps_paths):
        # Load
        with open(path, 'rb') as f:
            key2laionindices = pickle.load(f)

        # Add to the inverse map
        for key, laionindices in key2laionindices.items():
            for laionindex in laionindices:
                if laionindex not in laionindex2keys:
                    laionindex2keys[laionindex] = []
                laionindex2keys[laionindex].append(key)

    print_verbose(f'\tfound {len(laionindex2keys)} unique indices.')
    print_verbose('done!\n')

    # ----- Drop samples with multiple labels -----
    print_verbose('dropping samples with multiplicity ...')

    drop_indices = []
    for laionindex, keys in laionindex2keys.items():
        if len(keys) > 1:
            drop_indices.append(laionindex)

    print_verbose(f'\tfound {len(drop_indices)} samples with multiple labels.')

    for drop_idx in drop_indices:
        laionindex2keys.pop(drop_idx)

    print_verbose('done!\n')

    # ----- Load lemma to wnid map -----
    with open(params['lemma2wnid_path'], 'rb') as f:
        lemma2wnid = pickle.load(f)

    # ----- Design queries -----
    laionindex2query = {}
    for laionindex, keys in laionindex2keys.items():
        assert len(keys) == 1
        key = keys[0]

        wnid = lemma2wnid.get(key, key)

        laionindex2query[laionindex] = query_func(wnid)

    # Add to df
    df[query_col] = df.index.map(laionindex2query)

    # ----- Init. CLIP -----
    clip = CLIP()

    # ----- Loop over keys ------
    laionindices = list(laionindex2query.keys())
    for cnt in tqdm(range(0, len(laionindices), configs.CLIPConfig.BATCH_SIZE),
                    desc='calc. clip text to query similarity', disable=not logu.verbose):
        indices_batch = laionindices[cnt: (cnt + configs.CLIPConfig.BATCH_SIZE)]

        # Extract a batch
        texts_batch = \
            df.loc[indices_batch, configs.LAIONConfig.TEXT_COL].fillna(configs.CLIPConfig.REPLACE_NA_STR).tolist()
        queries_batch = df.loc[indices_batch, query_col].fillna(configs.CLIPConfig.REPLACE_NA_STR).tolist()

        try:
            # Get embeddings
            text_embeds_batch = clip.text_embeds(texts_batch)
            text_embeds_batch_norm = normalize(text_embeds_batch, axis=1, norm='l2')

            query_embeds_batch = clip.text_embeds(queries_batch)
            query_embeds_batch_norm = normalize(query_embeds_batch, axis=1, norm='l2')

            # Find similarities
            sims = np.sum(text_embeds_batch_norm * query_embeds_batch_norm, axis=1).tolist()

            # Update df
            df.loc[indices_batch, sim_col] = sims

        except Exception as e:
            print(str(e))

    # ----- Save -----
    print_verbose('saving ...')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
