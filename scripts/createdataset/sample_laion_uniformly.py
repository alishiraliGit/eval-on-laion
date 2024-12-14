import sys
import os
import argparse
import pickle
import glob

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


def sample_uniform(x):
    if len(x) <= configs.LAIONSamplingConfig.UNIFORM_SAMPLES:
        return sorted(x)
    else:
        np.random.seed(42)  # Fix the seed for reproducibility
        return sorted(np.random.choice(x, size=configs.LAIONSamplingConfig.UNIFORM_SAMPLES, replace=False).tolist())


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

    parser.add_argument('--load_file_name', type=str, default='lemma2laionindices(substring_matched*).pkl')

    parser.add_argument('--save_file_name', type=str, default='lemma2uniformlaionindices(substring_matched).pkl')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=False)

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Load and merges labels -----
    labels_paths = glob.glob(os.path.join(params['labels_path'], params['load_file_name']))

    print_verbose(f'found {len(labels_paths)} files.\n')
    print_verbose('\n'.join(labels_paths) + '\n')

    key2laionindices = {}
    for path in tqdm(labels_paths, desc='loading and merging labels'):
        # Load
        with open(path, 'rb') as f:
            key2laionindices_i = pickle.load(f)

        # Merge
        for key, laionindices in key2laionindices_i.items():
            if key not in key2laionindices:
                key2laionindices[key] = laionindices
            else:
                key2laionindices[key].extend(laionindices)

    # ----- Sample -----
    # Uniform samples
    key2uniformlaionindices = {}
    for key, laionindices in tqdm(key2laionindices.items(), desc='uniform sampling'):
        key2uniformlaionindices[key] = sample_uniform(laionindices)

    # ----- Save -----
    print_verbose('saving ...')

    with open(os.path.join(params['labels_path'], params['save_file_name']), open_type) as f:
        pickle.dump(key2uniformlaionindices, f)

    print_verbose('done!\n')
