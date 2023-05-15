import sys
import os
import json
import argparse
import glob
import pickle
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str,
                        default=os.path.join('imagenetv2', 'imagenetv2-matched-frequency'))
    parser.add_argument('--class_info_path', type=str, default=os.path.join('imagenetv2', 'class_info.json'))

    parser.add_argument('--dataframe_save_path', type=str, default=os.path.join('imagenetv2'))
    parser.add_argument('--labels_save_path', type=str, default=os.path.join('imagenetv2', 'processed', 'labels'))

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Path
    os.makedirs(params['dataframe_save_path'], exist_ok=True)
    os.makedirs(params['labels_save_path'], exist_ok=True)

    # Naming
    prefix = os.path.split(params['images_path'])[1]
    df_file_name = prefix + '.parquet'

    print_verbose('done!\n')

    # ----- Load class info -----
    print_verbose('loading and process class info ...')

    # Load
    with open(params['class_info_path'], 'rb') as f:
        class_info = json.load(f)

    # Process
    cid2wnid = {}
    for row in class_info:
        cid2wnid[row['cid']] = row['wnid']

    print_verbose('done!\n')

    # ----- Go over images -----
    print_verbose('going over images ...')

    imagename2wnid = {}
    for cid, wnid in tqdm(cid2wnid.items()):
        image_paths = glob.glob(os.path.join(params['images_path'], str(cid), '*'))
        print_verbose(f'\tfound {len(image_paths)} images for cid {cid}.')

        for image_path in image_paths:
            imagename = os.path.join(str(cid), os.path.split(image_path)[1])
            imagename2wnid[imagename] = wnid

    print_verbose('done!\n')

    # ----- Save ------
    # Dataframe
    print_verbose('saving dataframe ....')

    df = pd.DataFrame(
        index=list(imagename2wnid.keys())
    )

    df.to_parquet(os.path.join(params['dataframe_save_path'], df_file_name), index=True)

    print_verbose('done!\n')

    # Labels
    print_verbose('saving labels ....')

    with open(os.path.join(params['labels_save_path'], f'{prefix}-imagename2wnid.pkl'), 'wb') as f:
        pickle.dump(imagename2wnid, f)

    print_verbose('done!\n')
