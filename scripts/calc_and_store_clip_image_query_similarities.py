import sys
import os
import multiprocessing
import time
import argparse
import pickle
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.retrieve_image import download_image_content, verify_image
from core.clip import CLIP
from core.queries import select_queries, QueryType


def download_image_wrapper(args):
    idx, row = args
    try:
        img_content = download_image_content(row[configs.LAIONConfig.URL_COL])
        verify_image(img_content)
        return idx, img_content, None
    except Exception as e:
        return idx, None, {'cause': f'In downloading image of index {idx} an error occurred.', 'error': e}


def calc_image_to_text_similarities(idx_img_txt_list, clip_mdl: CLIP):
    # Load the images
    inds = []
    imgs = []
    txts = []
    errs = []
    for idx, img_content, txt in idx_img_txt_list:
        try:
            img = Image.open(BytesIO(img_content))

            inds.append(idx)
            imgs.append(img)
            txts.append(txt)

        except Exception as e:
            errs.append({'cause': f'In loading image of index {idx} from image content an error occurred.', 'error': e})

    # Calc. similarities
    try:
        sims = clip_mdl.similarities(texts=txts, images=imgs)
        return inds, sims, errs
    except Exception as e:
        errs.append({'cause': 'In calc. image-text similarities an error occurred.', 'error': e})
        return inds, [np.nan]*len(inds), errs


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_file_name', type=str, default='wnid2laionindices(substring_matched).pkl')

    # Method
    parser.add_argument('--substring_matched_filtered', action='store_true')
    parser.add_argument('--substring_matched_filtered_most_similar_images', action='store_true')
    parser.add_argument('--ilsvrc_val_most_similar_images', action='store_true')
    parser.add_argument('--imagenet_captions_most_similar_text_to_texts', action='store_true')
    parser.add_argument('--queried', action='store_true')

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=6)

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--save_freq', type=int, default=1000)

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Compute
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Set the files prefix
    if params['substring_matched_filtered']:
        prefix = configs.LAIONConfig.SUBSET_SM_FILTERED_PREFIX
    elif params['substring_matched_filtered_most_similar_images']:
        prefix = configs.LAIONConfig.SUBSET_SM_FILTERED_MOST_SIMILAR_IMG_IMG_PREFIX
    elif params['ilsvrc_val_most_similar_images']:
        prefix = configs.LAIONConfig.SUBSET_VAL_MOST_SIMILAR_IMG_IMG_PREFIX
    elif params['imagenet_captions_most_similar_text_to_texts']:
        prefix = configs.LAIONConfig.SUBSET_IC_MOST_SIMILAR_TXT_TXT_PREFIX
    elif params['queried']:
        prefix = configs.LAIONConfig.SUBSET_QUERIED_PREFIX
    else:
        raise Exception('Unknown method!')

    # Query
    query_func = select_queries([params['query_type']])[0]

    # Column names
    query_col = params['query_type'] + '_' + 'wnid'
    image_to_query_sim_col = f'image_to_{query_col}_similarity'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Loading labels -----
    print_verbose('loading labels ...')

    with open(os.path.join(params['labels_path'], params['labels_file_name']), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Find the inverse map -----
    print_verbose('finding inverse map ...')

    laionindex2wnids = {}
    for wnid, laion_indices in wnid2laionindices.items():
        for laion_idx in laion_indices:
            if laion_idx not in laionindex2wnids:
                laionindex2wnids[laion_idx] = []
            laionindex2wnids[laion_idx].append(wnid)

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + file_name_wo_prefix
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose('done!\n')

    # ----- Preprocess -----
    print_verbose('preprocess ...')

    # Drop rows with multiple labels
    selected_laion_indices = [laion_idx for laion_idx, wnids in laionindex2wnids.items() if len(wnids) == 1]

    df = df.loc[selected_laion_indices]

    # Create a new column
    if image_to_query_sim_col not in df:
        df[image_to_query_sim_col] = np.nan

    # Find rows w/o similarity
    df_todo = df.iloc[np.isnan(df[image_to_query_sim_col].tolist())]

    print_verbose('done!\n')

    # ----- Design queries -----
    print_verbose('design queries ...')

    laionindex2query = {}
    for laionindex, wnids in laionindex2wnids.items():
        if len(wnids) > 1:
            continue

        wnid = wnids[0]

        laionindex2query[laionindex] = query_func(wnid)

    # Add to df
    df[query_col] = df.index.map(laionindex2query)

    print_verbose('done!\n')

    # ----- Init. parallel download -----
    pool_download = multiprocessing.Pool(params['n_process_download'])

    # ----- Start download -----
    download_results = pool_download.imap(download_image_wrapper, df_todo.iterrows())

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    download_ready_results = []
    errors = []

    i_batch = 0
    for i_res, down_res in tqdm(enumerate(download_results), desc='download and calc. sim.', total=len(df_todo)):
        # Catch the errors in downloading
        index, image_content, error = down_res

        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
        else:
            # Get the queries
            text = df.loc[index, query_col]

            # Append the downloaded image to the batch
            download_ready_results.append([index, image_content, text])

        if len(download_ready_results) < configs.CLIPConfig.BATCH_SIZE and i_res < (len(df_todo) - 1):
            continue
        if len(download_ready_results) == 0:
            continue

        # Calc. embeddings
        indices_batch, similarities_batch, errors_batch = calc_image_to_text_similarities(download_ready_results, clip)

        for error in errors_batch:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Update df
        df.loc[indices_batch, image_to_query_sim_col] = similarities_batch

        # Save
        if ((i_batch + 1) % params['save_freq'] == 0) or i_res == (len(df_todo) - 1):
            print_verbose('saving ....')

            df.to_parquet(subset_file_path, index=True)

            print_verbose('done!\n')

        # Empty current batch
        download_ready_results = []
        i_batch += 1

    # ----- Close progress bars and processes -----
    pool_download.close()
    pool_download.join()

    time.sleep(1)

    # ----- Save error logs ------
    print_verbose('saving error logs ....')

    err_file_path = subset_file_path.replace('parquet', 'imgquerysim_errors.txt')
    with open(err_file_path, 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')

    print_verbose('saving ....')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
