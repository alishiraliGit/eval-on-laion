import sys
import os
import argparse
import pickle
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from core.clip import CLIP
from core.queries import select_queries, QueryType


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))

    parser.add_argument('--imagename2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'imagename2wnid.pkl'))

    parser.add_argument('--save_path', type=str, default=os.path.join('ilsvrc2012', 'processed'))

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Compute
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Query
    query_func = select_queries([params['query_type']])[0]

    # Column names
    query_col = params['query_type'] + '_' + 'wnid'
    image_to_query_sim_col = f'image_to_{query_col}_similarity'

    # Further naming
    imageindex2imagename = lambda image_idx: 'ILSVRC2012_val_%08d.JPEG' % image_idx

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load labels -----
    print_verbose('loading labels ...')

    with open(os.path.join(params['imagename2wnid_path']), 'rb') as f:
        imagename2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load and calc. similarity -----
    all_indices = []
    all_queries = []
    all_similarities = []

    images_batch = []
    indices_batch = []
    for idx in tqdm(range(1,  configs.ILSVRCConfigs.NUM_VAL + 1)):
        # Load the image
        image = Image.open(os.path.join(params['images_path'], imageindex2imagename(idx)))

        if image.mode == 'RGB':
            images_batch.append(image)
            indices_batch.append(idx)

        if len(images_batch) < configs.ILSVRCPredictorsConfig.BATCH_SIZE and idx < configs.ILSVRCConfigs.NUM_VAL:
            continue

        if len(images_batch) == 0:
            continue

        # Design queries
        queries_batch = [query_func(imagename2wnid[imageindex2imagename(idx)]) for idx in indices_batch]

        # Find image-to-query similarities
        similarities_batch = clip.similarities(texts=queries_batch, images=images_batch)

        # Step
        all_indices.extend(indices_batch)
        all_queries.extend(queries_batch)
        all_similarities.extend(similarities_batch)

        images_batch = []
        indices_batch = []

    # ----- Create a dataframe -----
    df = pd.DataFrame({query_col: all_queries, image_to_query_sim_col: all_similarities}, index=all_indices)

    # ----- Save error logs ------
    print_verbose('saving ....')

    save_file_name = 'ilsvrc_val_set_image_query_similarities.parquet'
    df.to_parquet(os.path.join(params['save_path'], save_file_name), index=True)

    print_verbose('done!\n')
