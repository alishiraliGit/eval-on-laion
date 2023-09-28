import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import utils
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_key', type=str, default='wnid', help='currently, only wnid supported.')  # TODO

    # Skimming
    parser.add_argument('--similarity_col', type=str)
    parser.add_argument('--similarity_th', type=float, default=0.58)
    parser.add_argument('--max_n_sample_per_key', type=int, default=50)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Set the files prefix
    prefix = params['prefix']

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + '_' + file_name_wo_prefix
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose(f'\tfound {len(df)} rows.')

    print_verbose('done!\n')

    # ----- Load labels (maps) -----
    print_verbose('loading labels ...')

    labels_file_name = f'{params["labels_key"]}2laionindices({prefix}).pkl'
    with open(os.path.join(params['labels_path'], labels_file_name), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Skimming -----
    df = df[df[params['similarity_col']] > params['similarity_th']]
    df = df[df['wnid_is_in_recognized_text'] == False]

    skimmed_wnid2laionindices = {}
    skimmed_laion_indices = []

    for wnid, laion_indices in tqdm(wnid2laionindices.items()):
        laion_indices = utils.intersect_lists(laion_indices, df.index)

        if len(laion_indices) < params['max_n_sample_per_key']:
            skimmed_wnid2laionindices[wnid] = laion_indices
        else:
            tq_sims = df.loc[laion_indices, params['similarity_col']]

            skimmed_wnid2laionindices[wnid] = \
                np.array(laion_indices)[np.argsort(tq_sims)[-params['max_n_sample_per_key']:]].tolist()

        skimmed_laion_indices.extend(skimmed_wnid2laionindices[wnid])

    df_skimmed = df.loc[skimmed_laion_indices]

    # ----- Save -----
    print_verbose('saving ...')

    prefix_skimmed = configs.NamingConfig.append_skimmed(prefix)

    # Save labels
    labels_file_name = f'{params["labels_key"]}2laionindices({prefix_skimmed}).pkl'

    with open(os.path.join(params['labels_path'], labels_file_name), open_type) as f:
        pickle.dump(skimmed_wnid2laionindices, f)

    # Save df
    subset_skimmed_file_name = prefix_skimmed + '_' + file_name_wo_prefix
    subset_skimmed_file_path = os.path.join(params['laion_path'], subset_skimmed_file_name)
    df_skimmed.to_parquet(subset_skimmed_file_path, index=True)

    print_verbose('done!\n')
