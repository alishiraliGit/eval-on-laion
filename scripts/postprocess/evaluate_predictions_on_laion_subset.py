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
from utils import utils
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu
from scripts.predict.download_and_predict import load_models


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_key', type=str, default='wnid', help='currently, only wnid supported.')  # TODO

    parser.add_argument('--predictions_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'ilsvrc_predictions'))
    parser.add_argument('--predictions_ver', type=str, default='*')

    # Predictors
    parser.add_argument('--predictors', type=str, default='selected')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Set the files prefix
    prefix = params['prefix']

    # Choose models
    model_names, _, _ = load_models(params['predictors'], do_init=False)

    print_verbose('\tusing models:\n\t' + '\n\t'.join(model_names))

    print_verbose('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + '_' + file_name_wo_prefix
    subset_pred_file_name = configs.NamingConfig.append_predicted(prefix) + '_' + file_name_wo_prefix

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

    labels_file_name = f'{params["labels_key"]}2laionindices({prefix}).pkl'
    with open(os.path.join(params['labels_path'], labels_file_name), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load predictions -----
    print_verbose('loading predictions ...')

    model2predictions = {}
    for model_name in model_names:
        pred_file_name = prefix + '_' + f'{model_name}_predictions_{params["predictions_ver"]}.csv'
        pred_file_paths = glob.glob(os.path.join(params['predictions_path'], pred_file_name))

        assert len(pred_file_paths) > 0, f'found no predictions for {model_name}.'

        print_verbose(f'\tfound {len(pred_file_paths)} prediction files for {model_name}.')

        pred_dfs = []
        for pred_file_path in pred_file_paths:
            print_verbose(f'\t\tloading {pred_file_path} ...')

            pred_df = pd.read_csv(
                pred_file_path,
                index_col=0,
                dtype={str(cnt): str for cnt in range(5)}
            )

            print_verbose(f'\t\tfound {len(pred_df)} rows.\n')

            pred_dfs.append(pred_df)

        pred_df_concat = pd.concat(pred_dfs)

        # Drop duplicate indices (keep only the first occurrence)
        pred_df_concat = pred_df_concat[~pred_df_concat.index.duplicated(keep='first')]

        print_verbose(f'\tfound overall {len(pred_df_concat)} unique rows.\n')

        model2predictions[model_name] = pred_df_concat

    print_verbose('done!\n')

    # ----- Find the inverse map -----
    print_verbose('finding all labels assigned to an example ...')

    laionindex2wnids = utils.find_inverse_map(wnid2laionindices)

    # Log
    avg_wnids_per_img = np.mean([len(wnids) for _, wnids in laionindex2wnids.items()])
    print_verbose('\ton average, %g possible classes per image.' % avg_wnids_per_img)

    unique_proportion = np.mean([len(wnids) == 1 for _, wnids in laionindex2wnids.items()])
    print_verbose('\ton average, %g of the images have a unique class.' % unique_proportion)

    print_verbose('done!\n')

    # ----- Evaluate predictions -----
    # Set column name
    top_k_col_name = lambda k, mdl: f'top_{k}_is_correct_{mdl}'

    # Big loop
    col2indices = {}
    col2values = {}
    for idx in tqdm(df.index):
        wnids = laionindex2wnids[idx]

        for model_name in model_names:
            pred_df = model2predictions[model_name]

            if idx not in pred_df.index:
                continue

            pred_list = pred_df.loc[idx].tolist()

            col = top_k_col_name(1, model_name)
            val = pred_list[0] in wnids
            if col not in col2indices:
                col2indices[col] = []
                col2values[col] = []
            col2indices[col].append(idx)
            col2values[col].append(val)

            col = top_k_col_name(5, model_name)
            val = np.any([p in wnids for p in pred_list])
            if col not in col2indices:
                col2indices[col] = []
                col2values[col] = []
            col2indices[col].append(idx)
            col2values[col].append(val)

    # ----- Add to dataframe -----
    for col, laion_indices in tqdm(col2indices.items(), desc='updating dataframe'):
        df.loc[laion_indices, col] = col2values[col]

    # ----- Save -----
    print_verbose('saving ...')

    df.to_parquet(os.path.join(params['laion_path'], subset_pred_file_name), index=True)

    print_verbose('done!\n')
