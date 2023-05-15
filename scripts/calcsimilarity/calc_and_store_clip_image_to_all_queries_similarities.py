import sys
import os
import multiprocessing
import time
import argparse
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from utils.ilsvrc_utils import load_lemmas_and_wnids
from core.clip import CLIP
from core.queries import select_queries, QueryType

from scripts.calcsimilarity.calc_and_store_clip_image_to_query_similarities import download_image_wrapper


def calc_image_to_queries_similarities(idx_img_list, q_embs_norm, clip_mdl: CLIP):
    # Load the images
    inds = []
    imgs = []
    errs = []
    for idx, img_content in idx_img_list:
        try:
            img = Image.open(BytesIO(img_content))

            inds.append(idx)
            imgs.append(img)

        except Exception as e:
            errs.append({'cause': f'In loading image of index {idx} from image content an error occurred.', 'error': e})

    # Calc. similarities
    try:
        img_embs = clip_mdl.image_embeds(imgs)
        img_embs_norm = normalize(img_embs, axis=1, norm='l2')
        sims = img_embs_norm.dot(q_embs_norm.T)

        return inds, sims, errs

    except Exception as e:
        errs.append({'cause': 'In calc. image-text similarities an error occurred.', 'error': e})
        return inds, np.ones((len(inds), len(q_embs_norm)))*np.nan, errs


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))

    # Method
    parser.add_argument('--method', type=str, help='Look at configs.LAIONConfig.')

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

    # Prefix/Postfix
    prefix = configs.LAIONConfig.method_to_prefix(params['method'])
    postfix = 'with_sims_to_all_queries'

    # Query
    query_func = select_queries([params['query_type']])[0]

    # Column names
    image_to_query_col_func = lambda w: f'image_to_{params["query_type"]}_{w}_similarity'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load synsets -----
    print_verbose('loading synsets ...')

    id_lemmas_df = load_lemmas_and_wnids(params['synsets_path'])
    all_wnids = id_lemmas_df[configs.ILSVRCConfigs.WNID_COL].tolist()

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + file_name_wo_prefix.replace('parquet', postfix + '.parquet')
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose('done!\n')

    # ----- Preprocess -----
    print_verbose('preprocess ...')

    # Create new columns
    for wnid in all_wnids:
        if image_to_query_col_func(wnid) not in df:
            df[image_to_query_col_func(wnid)] = np.nan

    # Find rows w/o similarity
    df_todo = df.iloc[np.isnan(df[image_to_query_col_func(all_wnids[-1])].tolist())]

    print_verbose('done!\n')

    # ----- Find the embeddings for queries -----
    print_verbose('calc. embeddings for the queries')

    queries = [query_func(wnid) for wnid in all_wnids]
    q_embeds = clip.text_embeds(queries)
    q_embeds_norm = normalize(q_embeds, axis=1, norm='l2')

    print_verbose('done!\n')

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
                                desc='download and calc. image to all queries sim.', total=len(df_todo)):
        # Catch the errors in downloading
        index, image_content, error = down_res

        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
        else:
            # Append the downloaded image to the batch
            download_ready_results.append([index, image_content])

        if len(download_ready_results) < configs.CLIPConfig.BATCH_SIZE and i_res < (len(df_todo) - 1):
            continue
        if len(download_ready_results) == 0:
            continue

        # Calc. similarities
        indices_batch, similarities_batch, errors_batch = \
            calc_image_to_queries_similarities(download_ready_results, q_embeds_norm, clip)

        for error in errors_batch:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Step
        all_indices.extend(indices_batch)
        all_similarities.extend(similarities_batch)

        # Save
        if ((i_batch + 1) % params['save_freq'] == 0) or i_res == (len(df_todo) - 1):
            print_verbose('saving ....')

            df.loc[all_indices, [image_to_query_col_func(wnid) for wnid in all_wnids]] = np.array(all_similarities)
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

    err_file_path = subset_file_path.replace('parquet', 'imgallqueriessim_errors.txt')
    with open(err_file_path, 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')

    print_verbose('saving ....')

    if len(all_indices) > 0:
        df.loc[all_indices, [image_to_query_col_func(wnid) for wnid in all_wnids]] = np.array(all_similarities)
        df.to_parquet(subset_file_path, index=True)
    else:
        print_verbose('\talready saved!')

    print_verbose('done!\n')
