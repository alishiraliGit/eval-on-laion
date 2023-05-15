import sys
import os
import multiprocessing
import time
import argparse
import pickle
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.clip import CLIP

from scripts.calcsimilarity.calc_and_store_intra_class_image_similarities import download_images_wrapper, df_gen


def calc_image_cross_similarities(inds, img_contents, local_imgs, clip_mdl: CLIP):
    # Load the images
    errs = []
    imgs = []
    success_inds = []
    for i_i, img_content in enumerate(img_contents):
        if img_content is None:
            continue

        try:
            img = Image.open(BytesIO(img_content))

            imgs.append(img)
            success_inds.append(inds[i_i])
        except Exception as e:
            errs.append(
                {
                    'cause': f'In loading image of index {inds[i_i]} from image content an error occurred.',
                    'error': e
                }
            )

    if len(imgs) == 0 or len(local_imgs) == 0:
        sims = []
        errs.append({'cause': 'No image for this class.', 'error': None})
        return success_inds, sims, errs

    # Calc. similarities
    try:
        embs = clip_mdl.image_embeds(imgs)
        embs = normalize(embs, axis=1, norm='l2')

        local_embs = clip_mdl.image_embeds(local_imgs)
        local_embs = normalize(local_embs, axis=1, norm='l2')

        sims = embs.dot(local_embs.T)
    except Exception as e:
        sims = []
        errs.append({'cause': 'In calc. image cross similarities an error occurred.', 'error': e})

    # Close the images
    for img in imgs:
        img.close()

    return success_inds, sims, errs


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_file_name', type=str)

    parser.add_argument('--save_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_image_similarities'))

    # Path to local data
    parser.add_argument('--local_images_path', type=str,
                        default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_train_selected'))
    parser.add_argument('--local_dataframe_path', type=str,
                        default=os.path.join('ilsvrc2012', 'imagenet_captions.parquet'))
    parser.add_argument('--local_index2filename_path', type=str, default=None)
    parser.add_argument('--local_index2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'labels', 'icimagename2wnid.pkl'))

    # Method
    parser.add_argument('--method', type=str, help='Look at configs.LAIONConfig.')

    # Sample
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--n_sample', type=int, default=100)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=6)

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Continue
    parser.add_argument('--no_continue', dest='continue', action='store_false')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Compute
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Set the prefixes
    prefix = configs.LAIONConfig.method_to_prefix(params['method'])
    local_prefix = os.path.split(params['local_dataframe_path'])[1].replace('.parquet', '_')

    # Saving
    os.makedirs(params['save_path'], exist_ok=True)
    wnid2savefilename = lambda w: prefix + 'to_' + local_prefix + f'img_img_sims({w}).pkl'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load LAION dataframe -----
    print_verbose('loading laion dataframe ...')

    # Load
    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + file_name_wo_prefix

    df = pd.read_parquet(os.path.join(params['laion_path'], subset_file_name))

    print_verbose('done!\n')

    # ----- Load local dataframe -----
    print_verbose('loading local dataframe ...')

    df_local = pd.read_parquet(params['local_dataframe_path'])

    print_verbose('done!\n')

    # ----- Load LAION labels -----
    print_verbose('loading laion labels ...')

    with open(os.path.join(params['labels_path'], params['labels_file_name']), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Sample LAION -----
    if params['do_sample']:
        print_verbose('sampling laion ...')

        wnid2laionindices = {wnid: laion_indices[:params['n_sample']] for
                             wnid, laion_indices in wnid2laionindices.items()}

        print_verbose('done!\n')

    # ----- Load local labels -----
    print_verbose('loading local labels ...')

    print_verbose('\tloading index2wnid ...')
    with open(os.path.join(params['local_index2wnid_path']), 'rb') as f:
        local_index2wnid = pickle.load(f)

    print_verbose('\tdone!\n')

    # Load the map to files
    print_verbose('\tloading index2filename map ...')

    if params['local_index2filename_path'] is None:
        print_verbose('\t\tdefaulting to identical map ...')
        local_index2filename = {idx: idx for idx in df_local.index}
    else:
        with open(params['local_index2filename_path'], 'rb') as f:
            local_index2filename = pickle.load(f)

    print_verbose('\tdone!\n')

    # Find the inverse map
    print_verbose('\tfinding the inverse map ...')

    wnid2filenames = {}
    for idx, wnid in local_index2wnid.items():
        file_name = local_index2filename[idx]

        if wnid not in wnid2filenames:
            wnid2filenames[wnid] = []
        wnid2filenames[wnid].append(file_name)

    print_verbose('\tdone!\n')

    print_verbose('done!\n')

    # ----- Init. parallel download -----
    pool_download = multiprocessing.Pool(params['n_process_download'])

    # ----- Start download -----
    download_results = pool_download.imap(download_images_wrapper, df_gen(wnid2laionindices, df, min_len=1))

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    errors = []

    for i_res, down_res in tqdm(enumerate(download_results),
                                desc='download and calc. cross image sim.', total=len(wnid2laionindices)):
        # Parse the downloaded images
        wnid, laion_indices, image_contents, down_errors = down_res

        if params['continue'] and os.path.exists(os.path.join(params['save_path'], wnid2savefilename(wnid))):
            continue

        for error in down_errors:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Load the local images
        if wnid not in wnid2filenames:
            continue

        local_images = []
        success_file_names = []
        for file_name in wnid2filenames[wnid]:
            file_path = os.path.join(params['local_images_path'], file_name)
            local_image = Image.open(file_path)

            if local_image.mode != 'RGB':
                local_image.close()
                continue

            local_images.append(local_image)
            success_file_names.append(file_name)

        # Calc. similarities
        success_laion_indices, similarities, sim_errors = \
            calc_image_cross_similarities(laion_indices, image_contents, local_images, clip)

        # Close the open local images
        for local_image in local_images:
            local_image.close()

        for error in sim_errors:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        if len(similarities) == 0:
            continue

        # Save similarities
        with open(os.path.join(params['save_path'], wnid2savefilename(wnid)), 'wb') as f:
            pickle.dump(
                {
                    'row_index': success_laion_indices,
                    'col_index': success_file_names,
                    'similarities': similarities
                },
                f
            )

    # ----- Close progress bars and processes -----
    pool_download.close()
    pool_download.join()

    time.sleep(1)

    # ----- Save error logs ------
    print_verbose('saving error logs ....')

    err_file_name = prefix + 'to_' + local_prefix + 'img_img_sims_errors.txt'
    with open(os.path.join(params['save_path'], err_file_name), 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')
