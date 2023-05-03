import sys
import os
import argparse
import pickle
import glob

from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

    parser.add_argument('--labels_filter', type=str, help='Can use * to include multiple files.')

    # Method
    parser.add_argument('--substring_matched', action='store_true')
    parser.add_argument('--substring_matched_filtered', action='store_true')
    parser.add_argument('--imagenet_captions_most_similar_text_to_texts', action='store_true')
    parser.add_argument('--queried', action='store_true')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false')

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=False)

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Load sampled indices -----
    file_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'found {len(file_paths)} key2laion indices files.\n')
    print_verbose('\n'.join(file_paths) + '\n')

    part2laionindices = {part: set() for part in range(configs.LAIONConfig.NUM_PARTS)}
    for path in tqdm(file_paths, desc='loading and merging sampled indices'):
        # Load
        with open(path, 'rb') as f:
            key2laionindices = pickle.load(f)

        # Add to corresponding part
        for _, laionindices in key2laionindices.items():
            for laionindex in laionindices:
                part, _ = laionu.imap_index(laionindex)

                part2laionindices[part].add(laionindex)

    # Log
    for part, laionindices in part2laionindices.items():
        print_verbose(f'part {part} has {len(laionindices)} samples.')

    # ----- Download and subset LAION -----
    part_dfs = []
    latest_part = -1
    for part, laionindices in tqdm(part2laionindices.items(), desc='downloading and subsetting'):
        if len(laionindices) == 0:
            continue
        latest_part = part

        # Load LAION part
        part_df = laionu.load_data_part(params['laion_path'], part, params['self_destruct'])

        # Subset
        part_df = part_df.loc[sorted(part2laionindices[part])]

        # Add
        part_dfs.append(part_df)

    # Concat part dfs
    df = pd.concat(part_dfs, axis=0)

    # ----- Save -----
    print_verbose('saving ...')

    if params['substring_matched']:
        prefix = configs.LAIONConfig.SUBSET_SM_PREFIX
    elif params['substring_matched_filtered']:
        prefix = configs.LAIONConfig.SUBSET_SM_FILTERED_PREFIX
    elif params['imagenet_captions_most_similar_text_to_texts']:
        prefix = configs.LAIONConfig.SUBSET_IC_MOST_SIMILAR_TXT_TXT_PREFIX
    elif params['queried']:
        prefix = configs.LAIONConfig.SUBSET_QUERIED_PREFIX
    else:
        raise Exception('Unknown method!')

    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, latest_part)
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    if os.path.exists(subset_file_path) and params['safe']:
        raise Exception('Subset already exists!')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
