import sys
import os
import argparse
import pickle
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from core.text_encoders import select_text_encoder
from core.queries import select_queries, QueryType, QueryKey
from utils import utils
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
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_filter', type=str, default='wnid2laionindices(substring_matched_part*).pkl')

    parser.add_argument('--lemma2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed',
                                             'lemma2wnid(unique_in_ilsvrc_ignored_empty_wnids).pkl'))

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)
    parser.add_argument('--query_key', type=str, help='wnid or lemma. Look at queries.QueryKey.')

    # Text encoder
    parser.add_argument('--text_encoder_ver', type=str, default=configs.CLIPConfig.DEFAULT_VERSION)

    # Compute
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_gpu', action='store_true')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--save_freq', type=int, default=1000)

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Prefix
    prefix = params['prefix']

    # Query
    query_func = select_queries([params['query_type']])[0]
    QueryKey.assert_query_key(params['query_key'])

    # Column names
    query_col = params['query_type'] + '_' + params['query_key']
    sim_col = f'text_to_{query_col}_similarity_{params["text_encoder_ver"]}'

    print_verbose('done!\n')

    # ----- Load the subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose(f'\tfound {len(df)} rows.')
    print_verbose('done!\n')

    # ----- Preprocess -----
    print_verbose('preprocess ...')

    # Find rows w/o similarity
    if sim_col not in df:
        df[sim_col] = np.nan
    df_todo = df.iloc[np.isnan(df[sim_col].tolist())]

    print_verbose('done!\n')

    # ----- Load maps and construct an inverse map -----
    print_verbose('loading labels (maps) and constructing an inverse map ...')

    maps_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'\tfound {len(maps_paths)} key2laion maps:\n')
    print_verbose('\t- ' + '\n\t- '.join(maps_paths))

    # Load maps
    maps = []
    for path in tqdm(maps_paths):
        with open(path, 'rb') as f:
            maps.append(pickle.load(f))

    # Find the inverse map
    laionindex2keys = utils.find_inverse_map(maps)

    print_verbose(f'\tfound {len(laionindex2keys)} unique indices.')
    print_verbose('done!\n')

    # ----- Drop samples with multiple labels -----
    print_verbose('dropping samples with multiplicity ...')

    drop_indices = utils.drop_keys_with_multiple_values(laionindex2keys)

    print_verbose(f'\tfound {len(drop_indices)} samples with multiple labels and dropped them.')

    print_verbose('done!\n')

    # ----- Load lemma to wnid map -----
    if params['query_key'] == QueryKey.LEMMA:
        with open(params['lemma2wnid_path'], 'rb') as f:
            lemma2wnid = pickle.load(f)

    # ----- Design queries -----
    laionindex2query = {}
    for laionindex, keys in laionindex2keys.items():
        assert len(keys) == 1
        key = keys[0]

        if params['query_key'] == QueryKey.WNID:
            wnid = key
            lemma = None
        elif params['query_key'] == QueryKey.LEMMA:
            wnid = lemma2wnid[key]
            lemma = key
        else:
            raise Exception(f'{params["query_key"]} is an invalid query_key!')

        laionindex2query[laionindex] = query_func(wnid, lemma)

    # Add to df
    df[query_col] = df.index.map(laionindex2query)

    # ----- Select the text encoder -----
    text_encoder, text_encoder_batch_size = select_text_encoder(params['text_encoder_ver'])

    # ----- Loop over keys ------
    laionindices = utils.intersect_lists(list(df_todo.index), list(laionindex2query.keys()))
    i_batch = 0
    for cnt in tqdm(range(0, len(laionindices), text_encoder_batch_size),
                    desc='calc. text to query similarity', disable=not logu.verbose):

        indices_batch = laionindices[cnt: (cnt + text_encoder_batch_size)]

        # Extract a batch
        texts_batch = df.loc[
            indices_batch, configs.LAIONConfig.TEXT_COL
        ].fillna(configs.TextEncoderConfig.REPLACE_NA_STR).tolist()
        queries_batch = df.loc[indices_batch, query_col].fillna(configs.TextEncoderConfig.REPLACE_NA_STR).tolist()

        # Get embeddings
        text_embeds_batch = text_encoder(texts_batch)
        query_embeds_batch = text_encoder(queries_batch)

        # Find similarities
        sims = utils.cosine_similarity(text_embeds_batch, query_embeds_batch).tolist()

        # Update df
        df.loc[indices_batch, sim_col] = sims

        # Save
        i_batch += 1
        if i_batch % params['save_freq'] == 0:
            print_verbose('saving ....')

            df.to_parquet(subset_file_path, index=True)

            print_verbose('done!\n')

    # ----- Save -----
    print_verbose('saving ...')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
