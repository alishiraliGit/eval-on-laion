import sys
import os
import argparse
import json
from tqdm import tqdm

from clip_retrieval.clip_client import ClipClient, Modality

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'from_clip_retrieval'))

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

    # ----- Init. the client -----
    client = ClipClient(
        url=configs.CLIPRetrievalConfig.BACKEND_URL,
        indice_name=params['clip_retrieval_index_name'],
        aesthetic_score=0,
        aesthetic_weight=0,
        modality=Modality.IMAGE,
        num_images=params['top_k'],
        deduplicate=True,
        use_safety_model=True,
        use_violence_detector=True
    )

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

    # ----- Retrieve -----
    for idx in tqdm(range(1, configs.ILSVRCConfigs.NUM_VAL + 1)):
        if idx in all_results:
            continue

        results = client.query(image=os.path.join(params['images_path'], 'ILSVRC2012_val_%08d.JPEG' % idx))

        all_results[idx] = results

        # Save
        if (idx % params['save_freq'] == 0) or (idx == configs.ILSVRCConfigs.NUM_VAL):
            print_verbose('saving ...')

            with open(os.path.join(params['save_path'], results_file_name), 'w') as f:
                json.dump(all_results, f)

            print_verbose('done!\n')
