import sys
import os
import json
import argparse
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
    parser.add_argument('--imagenet_captions_path', type=str,
                        default=os.path.join('ilsvrc2012', 'imagenet_captions.json'))
    parser.add_argument('--save_path', type=str, default=os.path.join('ilsvrc2012'))

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Naming
    prefix = 'imagenet_captions'
    df_file_name = prefix + '.parquet'

    print_verbose('done!\n')

    # ----- Load imagenet-captions -----
    print_verbose('loading imagenet-captions ...')

    with open(params['imagenet_captions_path'], 'rb') as f:
        imagenet_captions = json.load(f)

    print_verbose('done!\n')

    # ----- Loop over data -----
    texts = []
    image_names = []
    for i_ic, ic in tqdm(enumerate(imagenet_captions), desc='preprocess', total=len(imagenet_captions)):
        # Make a caption
        text = ' '.join([ic['title']] + ic['tags'] + [ic['description']])

        # Append
        texts.append(text)
        image_names.append(ic['filename'])

    # ----- Save ------
    print_verbose('saving ....')

    df = pd.DataFrame(
        {
            configs.LAIONConfig.TEXT_COL: texts,
        },
        index=image_names
    )

    df.to_parquet(os.path.join(params['save_path'], df_file_name), index=True)

    print_verbose('done!\n')
