import sys
import os
import pickle
import multiprocessing
import argparse
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from clip_retrieval.clip_client import ClipClient, Modality

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu


client = None


def init_client(clip_retrieval_index_name, top_k):
    global client
    client = ClipClient(
        url=configs.CLIPRetrievalConfig.BACKEND_URL,
        indice_name=clip_retrieval_index_name,
        aesthetic_score=0,
        aesthetic_weight=0,
        modality=Modality.IMAGE,
        num_images=top_k,
        deduplicate=True,
        use_safety_model=True,
        use_violence_detector=True
    )


def query(args):
    idx, row = args

    try:
        q_results = client.query(image=row[configs.LAIONConfig.URL_COL])
    except Exception as e:
        return -1, e

    return idx, q_results


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_file_name', type=str, default='wnid2laionindices(substring_matched).pkl')

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'from_clip_retrieval'))

    # Parse results
    parser.add_argument('--drop_identical', action='store_true')

    # Method
    parser.add_argument('--queried_clip_retrieval', action='store_true')
    parser.add_argument('--queried', action='store_true')

    # Sample
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--n_sample_per_wnid', type=int, default=20)
    parser.add_argument('--seed', type=int, default=111)

    # Multiprocessing
    parser.add_argument('--n_process', type=int, default=6)

    # CLIP retrieval
    parser.add_argument('--clip_retrieval_index_name', type=str, default='laion_400m',
                        help='laion5B-L-14, laion5B-H-14, or laion_400m')

    # Size
    parser.add_argument('--top_k', type=int, default=50)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--save_freq', type=int, default=1000)

    # Continue
    parser.add_argument('--no_continue', dest='continue', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Prefix
    if params['queried_clip_retrieval']:
        prefix = configs.LAIONConfig.SUBSET_VAL_MOST_SIMILAR_IMG_IMG_PREFIX
    elif params['queried']:
        prefix = configs.LAIONConfig.SUBSET_QUERIED_PREFIX
    else:
        prefix = configs.LAIONConfig.SUBSET_SM_FILTERED_PREFIX

    print('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, params['laion_until_part'])

    df = pd.read_parquet(os.path.join(params['laion_path'], subset_file_name))

    print_verbose(f'\tdata has {len(df)} rows.')
    print_verbose('done!\n')

    # ----- Sample -----
    if params['do_sample']:
        print_verbose('sampling ...')

        # Load labels
        with open(os.path.join(params['labels_path'], params['labels_file_name']), 'rb') as f:
            wnid2laionindices = pickle.load(f)

        # Sample per wnid
        np.random.seed(params['seed'])

        wnid2sampledlaionindices = {}
        for wnid, laion_indices in wnid2laionindices.items():
            if len(laion_indices) == 0:
                continue

            n_s = np.minimum(params['n_sample_per_wnid'], len(laion_indices))

            sampled_laion_indices = np.random.choice(laion_indices, size=n_s, replace=False)

            wnid2sampledlaionindices[wnid] = sampled_laion_indices

        # Subset df
        subset_indices = set()
        for wnid, sampled_laion_indices in wnid2sampledlaionindices.items():
            subset_indices.update(sampled_laion_indices)

        df = df.loc[list(subset_indices)]

        print_verbose(f'\tsampled from {len(wnid2sampledlaionindices)} wnids.')
        print_verbose(f'\tsampled data has {len(df)} rows.')
        print_verbose('done!\n')

    # ----- Load previous results (if any) -----
    results_file_name = f'top{params["top_k"]}_{prefix}most_similars_from_{params["clip_retrieval_index_name"]}.json'

    if os.path.exists(os.path.join(params['save_path'], results_file_name)) and params['continue']:
        print_verbose('loading previous results to continue ...')

        with open(os.path.join(params['save_path'], results_file_name), 'r') as f:
            all_results = json.load(f)

        # Convert str keys to int
        all_results = {int(k): v for k, v in all_results.items()}

        print_verbose(f'\tfound {len(all_results)} rows!')

        print_verbose('done!\n')
    else:
        all_results = {}

    # ----- Select images -----
    todo_indices = [index for index in df.index if index not in all_results]
    df = df.loc[todo_indices]

    # ----- Init. a pool -----
    pool = multiprocessing.Pool(
        params['n_process'],
        initializer=init_client,
        initargs=(params['clip_retrieval_index_name'], params['top_k'])
    )

    # ----- Start the pool -----
    pool_results = pool.imap(query, df.iterrows())

    # ----- Retrieve -----
    for i_res, result in tqdm(enumerate(pool_results), total=len(todo_indices)):
        index, query_results = result

        # Catch the error
        if index < 0:
            print_verbose(f'error caught in fetching the results: \n {str(query_results)}')
            continue
        if isinstance(query_results, dict):
            continue

        # Drop identical
        if params['drop_identical']:
            url = df.loc[index, configs.LAIONConfig.URL_COL]
            query_results = [query_res for query_res in query_results
                             if query_res[configs.CLIPRetrievalConfig.URL_COL] != url]

        # Store
        all_results[index] = query_results

        # Save
        if (i_res + 1) % params['save_freq'] == 0:
            print_verbose('saving ...')

            with open(os.path.join(params['save_path'], results_file_name), 'w') as f:
                json.dump(all_results, f)

            print_verbose('done!\n')

    # Save final results
    print_verbose('saving ...')

    with open(os.path.join(params['save_path'], results_file_name), 'w') as f:
        json.dump(all_results, f)

    print_verbose('done!\n')
