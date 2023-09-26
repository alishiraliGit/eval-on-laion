import sys
import os
import argparse
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils.ilsvrc_utils import load_lemmas_and_wnids
from core.queries import QueryType


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--load_path', type=str, default=os.path.join('ilsvrc2012'))
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--index2filename_path', type=str, default=None)

    parser.add_argument('--synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))

    parser.add_argument('--labels_file_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'labels', 'imagename2wnid.pkl'))

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)

    # CLIP version
    parser.add_argument('--clip_ver', type=str, default=configs.CLIPConfig.DEFAULT_VERSION)

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
    top_k_col_func = lambda k: f'top_{k}_is_correct_{params["clip_ver"]}'
    image_to_query_col_func = lambda w: f'image_to_{params["query_type"]}_{w}_similarity_{params["clip_ver"]}'

    print_verbose('done!\n')

    # ----- Load synsets -----
    print_verbose('loading synsets ...')

    id_lemmas_df = load_lemmas_and_wnids(params['synsets_path'])
    all_wnids = id_lemmas_df[configs.ILSVRCConfigs.WNID_COL].tolist()

    image_to_query_sim_cols = [image_to_query_col_func(w) for w in all_wnids]

    print_verbose('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading df ...')

    df_file_name = prefix + '.parquet'
    df = pd.read_parquet(os.path.join(params['load_path'], df_file_name))

    print_verbose('done!\n')

    # ----- Load labels (maps) -----
    print_verbose('loading labels ...')

    with open(params['labels_file_path'], 'rb') as f:
        index2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load sims. to all queries and evaluating -----
    print_verbose('finding files with sims. to all queries ...')

    df_with_sims_file_name = \
        configs.NamingConfig.append_with_sims_to_all_queries(prefix + '*', params['clip_ver']) + '.parquet'

    df_with_sims_file_paths = glob.glob(os.path.join(params['load_path'], df_with_sims_file_name))

    print_verbose(f'found {len(df_with_sims_file_paths)} files.')

    for df_with_sims_file_path in df_with_sims_file_paths:
        print_verbose(f'\tloading {df_with_sims_file_path} ...')

        df_with_sims = pd.read_parquet(df_with_sims_file_path)

        print_verbose('\tdone!')

        # Evaluation
        col2indices = {}
        col2values = {}

        for idx, row in tqdm(df_with_sims.iterrows(), desc='evaluating'):
            wnid = index2wnid[idx]

            if wnid not in all_wnids:  # Happens only for one WNID
                for top_k in [1, 5]:
                    col = top_k_col_func(top_k)
                    val = None
                    if col not in col2indices:
                        col2indices[col] = []
                        col2values[col] = []
                    col2indices[col].append(idx)
                    col2values[col].append(val)
                continue

            true_col = image_to_query_col_func(wnid)
            true_iq_sim = row[true_col]

            iq_sims = np.sort(row[image_to_query_sim_cols].tolist())

            for top_k in [1, 5]:
                col = top_k_col_func(top_k)
                val = true_iq_sim >= iq_sims[-top_k]
                if col not in col2indices:
                    col2indices[col] = []
                    col2values[col] = []
                col2indices[col].append(idx)
                col2values[col].append(val)

        # Add to dataframe
        for col, laion_indices in tqdm(col2indices.items(), desc='updating dataframe'):
            df.loc[laion_indices, col] = col2values[col]

    # ----- Save -----
    print_verbose('saving ...')

    df.to_parquet(os.path.join(params['load_path'], df_file_name), index=True)

    print_verbose('done!\n')
