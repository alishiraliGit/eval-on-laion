import sys
import os
import argparse
import glob
import pandas as pd
from tqdm.auto import tqdm
import gc

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--src_prefix', type=str, help='Can use *.')
    parser.add_argument('--target_prefix', type=str, help='Should be unique.')

    # Column
    parser.add_argument('--column', type=str)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=False)

    # Prefixes
    src_prefix = params['src_prefix']
    target_prefix = params['target_prefix']

    # Column names
    col = params['column']

    print_verbose('done!\n')

    # ----- Load the target -----
    print_verbose('loading target ...')

    target_file_name = target_prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    target_file_path = os.path.join(params['laion_path'], target_file_name)

    target_df = pd.read_parquet(target_file_path)

    print_verbose(f'\tfound {len(target_df)} rows.')
    print_verbose('done!\n')

    # ----- Find the source(s) -----
    print_verbose('finding source(s) ...')

    src_paths = glob.glob(os.path.join(params['laion_path'], params['src_prefix']))

    print_verbose(f'\tfound {len(src_paths)} source(s):\n')
    print_verbose('\t- ' + '\n\t- '.join(src_paths))

    print_verbose('done!\n')

    # ----- Load and copy column -----
    print_verbose(f'loading sources and copying column {col} ...')

    for path in tqdm(src_paths):
        src_df = pd.read_parquet(path)

        target_df.loc[src_df.index, col] = src_df[col]

        del src_df
        gc.collect()

    print_verbose('done!\n')

    # ----- Save error logs ------
    print_verbose('saving ...')

    target_df.to_parquet(target_file_path, index=True)

    print_verbose('done!\n')
