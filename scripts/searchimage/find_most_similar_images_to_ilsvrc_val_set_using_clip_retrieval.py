import sys
import os
import multiprocessing
import argparse
import json
from tqdm import tqdm

from clip_retrieval.clip_client import ClipClient, Modality

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


def init_client(clip_retrieval_index_name, top_k):
    return ClipClient(
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


def init_client_and_retrieve(args):
    idx, pars = args

    clt = init_client(pars['clip_retrieval_index_name'], pars['top_k'])

    try:
        q_results = clt.query(image=os.path.join(pars['images_path'], 'ILSVRC2012_val_%08d.JPEG' % idx))
    except Exception as e:
        return -1, e

    return idx, q_results


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'from_clip_retrieval'))

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

    os.makedirs(params['save_path'], exist_ok=True)

    # ----- Load previous results (if any) -----
    results_file_name = f'top{params["top_k"]}_val_most_similars_from_{params["clip_retrieval_index_name"]}.json'

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
    todo_indices = [index for index in range(1, configs.ILSVRCConfigs.NUM_VAL + 1) if index not in all_results]

    # ----- Init. a pool -----
    pool = multiprocessing.Pool(params['n_process'])

    # ----- Start the pool -----
    pool_results = pool.imap(init_client_and_retrieve, [(index, params) for index in todo_indices])

    # ----- Retrieve -----
    for i_res, result in tqdm(enumerate(pool_results), total=len(todo_indices)):
        index, query_results = result

        # Catch the error
        if index < 0:
            print_verbose(f'error caught in fetching the results: \n {str(query_results)}')
            continue

        all_results[index] = query_results

        # Save
        if ((i_res + 1) % params['save_freq'] == 0) or ((i_res + 1) == len(todo_indices)):
            print_verbose('saving ...')

            with open(os.path.join(params['save_path'], results_file_name), 'w') as f:
                json.dump(all_results, f)

            print_verbose('done!\n')
