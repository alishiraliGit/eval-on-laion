import sys
import os
import argparse
import subprocess
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu
from core.queries import QueryType


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))

    # Chunks
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--from_chunk', type=int, default=1)

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)

    # CLIP version
    parser.add_argument('--clip_ver', type=str, default=configs.CLIPConfig.DEFAULT_VERSION)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=6)

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

    verbose = params.pop('verbose')

    # Prefix
    prefix = params['prefix']

    # Chunk size
    ch_size = params.pop('chunk_size')

    # Compute
    no_gpu = params.pop('no_gpu')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + '_' + file_name_wo_prefix

    df_all = pd.read_parquet(os.path.join(params['laion_path'], subset_file_name))

    print_verbose('done!\n')

    # ----- Loop over chunks -----
    for ch, ch_start in enumerate(range(0, len(df_all), ch_size)):
        if ch + 1 < params['from_chunk']:
            continue

        print_verbose(f'chunk {ch + 1} ...')

        ch_prefix = prefix + f'_chunk{ch + 1}'
        subset_ch_file_name = ch_prefix + '_' + file_name_wo_prefix
        subset_ch_file_path = os.path.join(params['laion_path'], subset_ch_file_name)

        if not os.path.exists(subset_ch_file_path):
            ch_df = df_all.iloc[ch_start: (ch_start + ch_size)]
            ch_df.to_parquet(subset_ch_file_path, index=True)

        params['prefix'] = ch_prefix

        ch_args = [f'--{k} {v}' for k, v in params.items()]
        ch_cmd = ['python', 'scripts/calcsimilarity/calc_and_store_clip_image_to_all_queries_similarities.py'] \
            + ' '.join(ch_args).split(' ')

        if no_gpu:
            ch_cmd.append('--no_gpu')
        if not verbose:
            ch_cmd.append('--no_verbose')

        subprocess.run(ch_cmd, text=True, check=True)
