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
from utils import laion_utils as laionu
from core.ilsvrc_predictors import model_names_1k, model_names_pt21k_ft1k, model_names_21k


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

    parser.add_argument('--predictions_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'ilsvrc_predictions'))
    parser.add_argument('--predictions_ver', type=str, default='*')

    # Method
    parser.add_argument('--queried_clip_retrieval', action='store_true')
    parser.add_argument('--queried', action='store_true')
    parser.add_argument('--query_type', type=str, default=None)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Boolean decisions
    queried = params['queried']
    queried_cr = params['queried_clip_retrieval']

    # Set the files prefix
    if queried_cr:
        prefix = configs.LAIONConfig.SUBSET_CLIP_RETRIEVAL_PREFIX
    elif queried:
        prefix = configs.LAIONConfig.SUBSET_QUERIED_PREFIX
    else:
        prefix = configs.LAIONConfig.SUBSET_PREFIX

    # Assertion
    if queried:
        assert params['query_type'] is not None

    # Choose models
    model_names = model_names_1k + model_names_pt21k_ft1k + model_names_21k
    map_models = [6, 5, 0, 1, 3, 2, 4]
    model_names = np.array(model_names)[map_models].tolist()

    print_verbose('\tusing models:\n\t' + '\n\t'.join(model_names))

    print_verbose('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + file_name_wo_prefix
    subset_pred_file_name = prefix + configs.LAIONConfig.PREDICTED_PREFIX + file_name_wo_prefix

    if os.path.exists(os.path.join(params['laion_path'], subset_pred_file_name)):
        print_verbose('\tfound a file already containing predictions:')
        print_verbose(f'\t{subset_pred_file_name}')

        df = pd.read_parquet(os.path.join(params['laion_path'], subset_pred_file_name))

    else:
        df = pd.read_parquet(os.path.join(params['laion_path'], subset_file_name))

    print_verbose(f'\tfound {len(df)} rows.')

    print_verbose('done!\n')

    # ----- Load labels (maps) -----
    print_verbose('loading labels ...')

    if queried_cr:
        map_file_name = 'wnid2crindices.pkl'
    elif queried:
        map_file_name = f'wnid2laionindices(query_{params["query_type"]}).pkl'
    else:
        map_file_name = 'wnid2uniformlaionindices(substring_matched).pkl'

    with open(os.path.join(params['labels_path'], map_file_name), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load predictions -----
    print_verbose('loading predictions ...')

    model2predictions = {}
    for model_name in model_names:
        pred_file_path = glob.glob(os.path.join(
            params['predictions_path'],
            prefix + f'{model_name}_predictions_{params["predictions_ver"]}.csv'))
        assert len(pred_file_path) == 1, 'found ' + '\n'.join(pred_file_path)
        pred_file_path = pred_file_path[0]

        print_verbose(f'\tloading {pred_file_path} ...')

        model2predictions[model_name] = pd.read_csv(
            pred_file_path,
            index_col=0,
            dtype={str(cnt): str for cnt in range(5)}
        )

        print_verbose(f'\tfound {len(model2predictions[model_name])} rows.\n')

    print_verbose('done!\n')

    # ----- Find the inverse map -----
    print_verbose('finding all labels assigned to an example ...')

    laionindex2wnids = {}
    for wnid, indices in wnid2laionindices.items():
        for idx in indices:
            if idx not in laionindex2wnids:
                laionindex2wnids[idx] = []
            laionindex2wnids[idx].append(wnid)

    # Log
    avg_wnids_per_img = np.mean([len(wnids) for _, wnids in laionindex2wnids.items()])
    print_verbose('\ton average, %g possible classes per image.' % avg_wnids_per_img)

    unique_proportion = np.mean([len(wnids) == 1 for _, wnids in laionindex2wnids.items()])
    print_verbose('\ton average, %g of the images have a unique class.' % unique_proportion)

    print_verbose('done!\n')

    # ----- Evaluate predictions -----
    # Set column name
    if queried_cr:
        top_k_col_name = lambda k, mdl: f'top_{k}_is_correct_cr_{mdl}'
    elif queried:
        top_k_col_name = lambda k, mdl: f'top_{k}_is_correct_{params["query_type"]}_{mdl}'
    else:
        top_k_col_name = lambda k, mdl: f'top_{k}_is_correct_{mdl}'

    # Big loop
    for idx in tqdm(df.index):
        if idx not in laionindex2wnids:
            continue

        wnids = laionindex2wnids[idx]

        for model_name in model_names:
            pred_df = model2predictions[model_name]

            if idx not in pred_df.index:
                continue

            pred_list = pred_df.loc[idx].tolist()

            df.loc[idx, top_k_col_name(1, model_name)] = pred_list[0] in wnids
            df.loc[idx, top_k_col_name(5, model_name)] = np.any([p in wnids for p in pred_list])

    # ----- Save -----
    print_verbose('saving ...')

    df.to_parquet(os.path.join(params['laion_path'], subset_pred_file_name), index=True)

    print_verbose('done!\n')