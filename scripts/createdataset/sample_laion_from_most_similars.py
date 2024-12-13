import sys
import os
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import utils
from utils import laion_utils as laionu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--load_prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_filter', type=str, default='wnid2laionindices(substring_matched_part*).pkl')
    parser.add_argument('--labels_key', type=str, default='wnid', help='wnid or lemma.')

    # Filtering
    parser.add_argument('--similarity_col', type=str)
    parser.add_argument('--similarity_th', type=float,
                        default=0.82, help='0.82 for CLIP, 0.97 for Bert, 0.58 for MPNet')

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

    # Prefix
    prefix = params['load_prefix']

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Load the subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose(f'\tfound {len(df)} rows.')
    print_verbose('done!\n')

    # ----- Remove NSFW -----
    if params['remove_nsfw']:
        print_verbose('removing nsfw ...')

        df.loc[df[configs.LAIONConfig.NSFW_COL] != configs.LAIONConfig.SAFE_TAG, params['similarity_col']] = np.nan

        print_verbose('done!\n')

    # ----- Load and join labels (maps) -----
    print_verbose('loading labels (maps) ...')

    maps_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'\tfound {len(maps_paths)} key2laion maps:\n')
    print_verbose('\t- ' + '\n\t- '.join(maps_paths))

    # Load maps
    maps = []
    for path in tqdm(maps_paths):
        with open(path, 'rb') as f:
            maps.append(pickle.load(f))

    key2laionindices = utils.join_maps(maps)

    print_verbose('done!\n')

    # ----- Sample -----
    print_verbose('sampling ...')

    # Choose indices
    all_laionindices = set()
    key2laionindices_sampled = {}
    for key, laionindices in tqdm(key2laionindices.items()):
        sims = np.array(df.loc[laionindices, params['similarity_col']].tolist())

        laionindices_sampled = np.array(laionindices)[sims > params['similarity_th']].tolist()

        key2laionindices_sampled[key] = laionindices_sampled

        all_laionindices.update(laionindices_sampled)

    all_laionindices = sorted(all_laionindices)

    # Subset
    df = df.loc[all_laionindices]

    print_verbose(f'\tsampled data has {len(df)} rows.')

    print_verbose('done!\n')

    # ----- Save -----
    print_verbose('saving ...')

    save_prefix = configs.NamingConfig.append_filtered(prefix, params['similarity_col'])

    # Save labels
    labels_file_name = f'{params["labels_key"]}2laionindices({save_prefix}).pkl'

    with open(os.path.join(params['labels_path'], labels_file_name), open_type) as f:
        pickle.dump(key2laionindices_sampled, f)

    # Save df
    sampled_subset_file_name = save_prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    sampled_subset_file_path = os.path.join(params['laion_path'], sampled_subset_file_name)

    if params['safe'] and os.path.exists(sampled_subset_file_path):
        raise FileExistsError

    df.to_parquet(sampled_subset_file_path, index=True)

    print_verbose('done!\n')
