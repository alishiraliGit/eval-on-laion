import sys
import os
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import laion_utils as laionu
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

    # Filtering
    parser.add_argument('--similarity_col', type=str, default='text_to_a_photo_of_name_def_wnid_similarity')
    parser.add_argument('--similarity_th', type=float)

    parser.add_argument('--remove_nsfw', action='store_true')

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
    logu.verbose = params['verbose']

    # Prefix
    prefix = configs.LAIONConfig.SUBSET_NOT_SAMPLED_PREFIX

    print_verbose('done!\n')

    # ----- Load the subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose(f'\tfound {len(df)} rows.')
    print_verbose('done!\n')

    # ----- Remove NSFW -----
    if params['remove_nsfw']:
        print_verbose('removing nsfw ...')

        df.loc[df[configs.LAIONConfig.NSFW_COL] != configs.LAIONConfig.SAFE_TAG, params['similarity_col']] = np.nan

        print_verbose('done!\n')

    # ----- Load labels (maps) -----
    print_verbose('loading labels (maps) ...')

    maps_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'\tfound {len(maps_paths)} key2laion maps:\n')
    print_verbose('\t- ' + '\n\t- '.join(maps_paths))

    key2laionindices = {}
    for path in tqdm(maps_paths):
        with open(path, 'rb') as f:
            key2laionindices_i = pickle.load(f)

        for key, laionindices in key2laionindices_i.items():
            if key not in key2laionindices:
                key2laionindices[key] = []
            key2laionindices[key].extend(laionindices)

    print_verbose('done!\n')

    # ----- Sample -----
    print_verbose('sampling ...')

    # Choose indices
    all_laionindices = set()
    for key, laionindices in tqdm(key2laionindices.items()):
        sims = np.array(df.loc[laionindices, params['similarity_col']].tolist())

        all_laionindices.update(np.array(laionindices)[sims > params['similarity_th']])

    all_laionindices = sorted(all_laionindices)

    # Subset
    df = df.loc[all_laionindices]

    print_verbose(f'\tsampled data has {len(df)} rows.')

    print_verbose('done!\n')

    # ----- Save -----
    print_verbose('saving ...')

    sampled_subset_file_name = configs.LAIONConfig.SUBSET_PREFIX \
        + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    sampled_subset_file_path = os.path.join(params['laion_path'], sampled_subset_file_name)

    if params['safe'] and os.path.exists(sampled_subset_file_path):
        raise FileExistsError

    df.to_parquet(sampled_subset_file_path, index=True)

    print_verbose('done!\n')
