import sys
import os
import argparse
import pandas as pd
import glob

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import utils
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu
from core.ilsvrc_predictors import select_ilsvrc_predictors


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--from_prefix', type=str, help='Look at configs.NamingConfig for conventions.')
    parser.add_argument('--to_prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--predictions_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'ilsvrc_predictions'))

    # Predictors
    parser.add_argument('--predictors', type=str, default='selected')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Prefix
    from_prefix = params['from_prefix']
    to_prefix = params['to_prefix']

    # Load model names
    model_names, _, _ = select_ilsvrc_predictors(params['predictors'], do_init=False)

    print_verbose('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading laion subsets ...')

    subset_file_name = to_prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose('done!\n')

    # ----- Load previous predictions in save_path ------
    print_verbose('loading previous predictions ...')

    for model_name in model_names:
        print_verbose(f'\tloading previous predictions of {model_name} ...')

        pred_file_name = from_prefix + '_' + f'{model_name}_predictions_*.csv'
        pred_file_paths = glob.glob(os.path.join(params['predictions_path'], pred_file_name))

        print_verbose(f'\tfound {len(pred_file_paths)} previous predictions.')

        for pred_file_path in pred_file_paths:
            print_verbose(f'\t\tloading {pred_file_path} ...')

            pred_df = pd.read_csv(
                pred_file_path,
                index_col=0,
                dtype={str(cnt): str for cnt in range(5)}
            )

            print_verbose(f'\t\tfound {len(pred_df)} rows.')

            common_indices = utils.intersect_lists(df.index, pred_df.index)

            print_verbose(f'\t\tfound {len(common_indices)} common indices.')

            if len(common_indices) > 0:
                print_verbose('\t\tsaving ...')

                to_pred_file_path = pred_file_path.replace(from_prefix, to_prefix)
                pred_df.loc[common_indices].to_csv(to_pred_file_path, index=True)

                print_verbose('\t\tdone!')
