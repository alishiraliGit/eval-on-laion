import sys
import os
import multiprocessing
import time
import argparse
import pickle
import glob
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import gc

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import utils
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.retrieve_image import download_image_content, verify_image
from core.foundationmodels.clip import CLIP
from core.queries import select_queries, QueryType, QueryKey


def download_image_wrapper(args):
    idx, row = args
    try:
        img_content = download_image_content(row[configs.LAIONConfig.URL_COL])
        verify_image(img_content)
        return idx, img_content, None
    except Exception as e:
        return idx, None, {'cause': f'In downloading image of index {idx} an error occurred.', 'error': str(e)}


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
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_filter', type=str, default='wnid2laionindices(substring_matched_part*).pkl')

    parser.add_argument('--lemma2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed',
                                             'lemma2wnid(unique_in_ilsvrc_ignored_empty_wnids).pkl'))

    # Subset
    parser.add_argument('--from_iloc', type=int, default=0)
    parser.add_argument('--to_iloc', type=int, default=-1)

    # Query
    parser.add_argument('--query_type', type=str, default=QueryType.NAME_DEF)
    parser.add_argument('--query_key', type=str, help='wnid or lemma. Look at queries.QueryKey.')

    # CLIP version
    parser.add_argument('--clip_ver', type=str, default=configs.CLIPConfig.DEFAULT_VERSION)

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

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Prefix
    prefix = params['prefix']

    # Query
    query_func = select_queries([params['query_type']])[0]
    QueryKey.assert_query_key(params['query_key'])

    # Column names
    query_col = params['query_type'] + '_' + params['query_key']
    image_to_query_sim_col = f'image_to_{params["query_type"]}_similarity_{params["clip_ver"]}'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP(ver=params['clip_ver'])

    print_verbose('done!\n')

    # ----- Load the subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    # Subset
    if params['to_iloc'] > 0:
        df = df.iloc[params['from_iloc']: params['to_iloc']]
    else:
        df = df.iloc[params['from_iloc']:]

    print_verbose(f'\tfound {len(df)} rows.')
    print_verbose('done!\n')

    # ----- Preprocess -----
    print_verbose('preprocess ...')

    # Find rows w/o similarity
    if image_to_query_sim_col not in df:
        df[image_to_query_sim_col] = np.nan
    df_todo = df.iloc[np.isnan(df[image_to_query_sim_col].tolist())]

    print_verbose('done!\n')

    # ----- Load maps and construct an inverse map -----
    print_verbose('loading labels (maps) and constructing an inverse map ...')

    maps_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'\tfound {len(maps_paths)} key2laion maps:\n')
    print_verbose('\t- ' + '\n\t- '.join(maps_paths))

    # Load maps
    maps = []
    for path in tqdm(maps_paths):
        with open(path, 'rb') as f:
            maps.append(pickle.load(f))

    # Find the inverse map
    laionindex2keys = utils.find_inverse_map(maps)

    print_verbose(f'\tfound {len(laionindex2keys)} unique indices.')
    print_verbose('done!\n')

    # ----- Drop samples with multiple labels -----
    print_verbose('dropping samples with multiplicity ...')

    drop_indices = utils.drop_keys_with_multiple_values(laionindex2keys)

    print_verbose(f'\tfound {len(drop_indices)} samples with multiple labels and dropped them.')

    print_verbose('done!\n')

    # ----- Load lemma to wnid map -----
    if params['query_key'] == QueryKey.LEMMA:
        with open(params['lemma2wnid_path'], 'rb') as f:
            lemma2wnid = pickle.load(f)

    # ----- Design queries -----
    laionindex2query = {}
    for laionindex, keys in laionindex2keys.items():
        assert len(keys) == 1
        key = keys[0]

        if params['query_key'] == QueryKey.WNID:
            wnid = key
            lemma = None
        elif params['query_key'] == QueryKey.LEMMA:
            wnid = lemma2wnid[key]
            lemma = key
        else:
            raise Exception(f'{params["query_key"]} is an invalid query_key!')

        laionindex2query[laionindex] = query_func(wnid, lemma)

    # Add to df
    df[query_col] = df.index.map(laionindex2query)

    # ----- Init. parallel download -----
    pool_download = multiprocessing.Pool(params['n_process_download'])

    # ----- Start download -----
    download_results = pool_download.imap(download_image_wrapper, df_todo.iterrows())

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    all_indices = []
    all_similarities = []

    download_ready_results = []
    errors = []

    i_batch = 0
    for i_res, down_res in tqdm(enumerate(download_results),
                                desc='download and calc. image to query sim.', total=len(df_todo)):
        # Catch the errors in downloading
        index, image_content, error = down_res

        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
        else:
            # Get the query
            text = df.loc[index, query_col]

            # Append the downloaded image to the batch
            download_ready_results.append([index, image_content, text])

        if len(download_ready_results) < configs.CLIPConfig.BATCH_SIZE and i_res < (len(df_todo) - 1):
            continue
        if len(download_ready_results) == 0:
            continue

        # Calc. similarities
        indices_batch, similarities_batch, errors_batch = calc_image_to_text_similarities(download_ready_results, clip)

        for error in errors_batch:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Step
        all_indices.extend(indices_batch)
        all_similarities.extend(similarities_batch)

        # Save
        if ((i_batch + 1) % params['save_freq'] == 0) or i_res == (len(df_todo) - 1):
            print_verbose('saving ....')

            df.loc[all_indices, image_to_query_sim_col] = all_similarities
            df.to_parquet(subset_file_path, index=True)

            all_indices = []
            all_similarities = []

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

    if len(all_indices) > 0:
        df.loc[all_indices, image_to_query_sim_col] = all_similarities
        df.to_parquet(subset_file_path, index=True)
    else:
        print_verbose('\talready saved!')

    print_verbose('done!\n')

    del df
    gc.collect()
