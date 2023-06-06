import sys
import os
import json
import argparse
import pickle
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--load_path', type=str, default=os.path.join('imagenet-captions', 'imagenet_captions.json'))
    parser.add_argument('--dataframe_path', type=str, default=os.path.join('imagenet-captions'))
    parser.add_argument('--labels_path', type=str, default=os.path.join('imagenet-captions', 'processed', 'labels'))

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Path
    os.makedirs(params['dataframe_path'], exist_ok=True)
    os.makedirs(params['labels_path'], exist_ok=True)

    # Naming
    prefix = 'imagenet_captions'
    df_file_name = prefix + '.parquet'

    print_verbose('done!\n')

    # ----- Load imagenet-captions -----
    print_verbose('loading imagenet-captions ...')

    with open(params['load_path'], 'rb') as f:
        imagenet_captions = json.load(f)

    print_verbose('done!\n')

    # ----- Loop over data -----
    texts = []
    image_names = []
    imagename2wnid = {}
    for i_ic, ic in tqdm(enumerate(imagenet_captions), desc='preprocess', total=len(imagenet_captions)):
        # Make a caption
        text = ' '.join([ic['title']] + ic['tags'] + [ic['description']])

        # Append
        texts.append(text)
        image_names.append(ic['filename'])

        # Read WNID
        wnid = ic['wnid']
        imagename2wnid[ic['filename']] = wnid

    # ----- Save ------
    # Dataframe
    print_verbose('saving dataframe ....')

    df = pd.DataFrame(
        {
            configs.LAIONConfig.TEXT_COL: texts,
        },
        index=image_names
    )

    # Drop duplicates
    df.index.name = 'ic_index'
    df = df.groupby('ic_index').first()

    df.to_parquet(os.path.join(params['dataframe_path'], df_file_name), index=True)

    print_verbose('done!\n')

    # Labels
    print_verbose('saving labels ....')

    with open(os.path.join(params['labels_path'], 'icimagename2wnid.pkl'), 'wb') as f:
        pickle.dump(imagename2wnid, f)

    print_verbose('done!\n')
