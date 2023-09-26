import sys
import os
import argparse
import glob
import pickle
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from scripts.predict.download_and_predict import load_models


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--load_path', type=str, default=os.path.join('ilsvrc2012'))
    parser.add_argument('--prefix', type=str)

    parser.add_argument('--labels_file_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'labels', 'imagename2wnid.pkl'))

    parser.add_argument('--predictions_path', type=str, default=os.path.join('ilsvrc2012', 'processed', 'predictions'))
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

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    df_file_name = prefix + '.parquet'
    df = pd.read_parquet(os.path.join(params['load_path'], df_file_name))

    print_verbose('done!\n')

    # ----- Load labels (maps) -----
    print_verbose('loading labels ...')

    with open(params['labels_file_path'], 'rb') as f:
        index2wnid = pickle.load(f)

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

    # ----- Evaluate predictions -----
    # Set column name
    top_k_col_name = lambda k, mdl: f'top_{k}_is_correct_{mdl}'

    # Big loop
    col2indices = {}
    col2values = {}
    for idx in tqdm(df.index):
        wnid = index2wnid[idx]

        for model_name in model_names:
            pred_df = model2predictions[model_name]

            if idx not in pred_df.index:
                continue

            pred_list = pred_df.loc[idx].tolist()

            col = top_k_col_name(1, model_name)
            val = wnid == pred_list[0]
            if col not in col2indices:
                col2indices[col] = []
                col2values[col] = []
            col2indices[col].append(idx)
            col2values[col].append(val)

            col = top_k_col_name(5, model_name)
            val = wnid in pred_list
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

    df.to_parquet(os.path.join(params['load_path'], df_file_name), index=True)

    print_verbose('done!\n')
