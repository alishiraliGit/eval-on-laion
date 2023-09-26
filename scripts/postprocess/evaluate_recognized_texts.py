import sys
import os
import argparse
import pickle
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import utils
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu
from core.queries import query_name


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_key', type=str, default='wnid', help='currently, only wnid supported.')  # TODO

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Set the files prefix
    prefix = params['prefix']

    # Set column name
    rec_text_col = 'recognized_text'
    is_in_col = f'{params["labels_key"]}_is_in_recognized_text'

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

    laionindex2wnids = utils.find_inverse_map(wnid2laionindices)

    print_verbose('done!\n')

    # ----- Evaluate predictions -----
    values = []
    for idx, row in tqdm(df.iterrows()):
        if row[rec_text_col] is None:
            values.append(False)
            continue

        rec_text = laionu.transform_text(row[rec_text_col])

        wnids = laionindex2wnids[idx]

        is_in_rec_text = False

        for wnid in wnids:
            wnid_text = laionu.transform_text(query_name(wnid))

            if wnid_text in rec_text:
                is_in_rec_text = True
                continue

        values.append(is_in_rec_text)

    # ----- Add to dataframe -----
    df[is_in_col] = values

    # ----- Save -----
    print_verbose('saving ...')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
