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


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--prefix', type=str, help='Can use *.')

    # Column
    parser.add_argument('--src_column', type=str)
    parser.add_argument('--target_column', type=str)

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
    prefix = params['prefix']

    # Column names
    src_col = params['src_column']
    target_col = params['target_column']

    print_verbose('done!\n')

    # ----- Load the subset and rename the column -----
    print_verbose('loading laion subset(s) and renaming their column ...')

    paths = glob.glob(os.path.join(params['laion_path'], params['prefix']))

    print_verbose(f'\tfound {len(paths)} subsets(s):\n')
    print_verbose('\t- ' + '\n\t- '.join(paths))

    for path in tqdm(paths):
        df = pd.read_parquet(path)

        df = df.rename(columns={src_col: target_col})

        df.to_parquet(path, index=True)

        del df
        gc.collect()

    print_verbose('done!\n')
