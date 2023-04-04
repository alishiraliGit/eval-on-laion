import sys
import os
import argparse
import numpy as np
import faiss
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import gdrive_utils as gdu


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--indices_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices', 'all_indices.npy'))

    parser.add_argument('--faiss_index_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'faiss_index', 'knn.index'))

    parser.add_argument('--indices_downloaded_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices_downloaded'))

    parser.add_argument('--embeddings_downloaded_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices_downloaded'))

    parser.add_argument('--credential_path', type=str,
                        default=os.path.join('credentials', configs.GDriveConfig.CRED_FILE_NAME))

    # Logging
    parser.add_argument('--verbose', type=bool, default=True)

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    # Env
    logu.verbose = params['verbose']
    os.makedirs(params['indices_downloaded_path'], exist_ok=True)
    os.makedirs(params['embeddings_downloaded_path'], exist_ok=True)

    # GDrive
    cred = gdu.authenticate(params['credential_path'])
    gdu.build_service(cred)

    # ----- Load indices already added to index -----
    # Load indices
    with open(params['indices_path'], 'rb') as f:
        # noinspection PyTypeChecker
        all_indices = np.load(f)

    # ----- Load the index -----
    print_verbose('loading the index ...')

    faiss_index = faiss.read_index(params['faiss_index_path'])

    print_verbose('done!')

    # ----- Get the list of indices and embeddings at GDrive -----
    indfileid2name = gdu.get_file_ids(configs.GDriveConfig.IND_FOLDER_ID)
    embfileid2name = gdu.get_file_ids(configs.GDriveConfig.EMB_FOLDER_ID)

    # ----- Determine which indices shall be downloaded -----
    tbd_indfileid2name = {}
    for ind_file_id, ind_file_name in indfileid2name.items():
        if not ind_file_name.startswith('indices'):
            continue

        if os.path.exists(os.path.join(params['indices_downloaded_path'], ind_file_name)):
            print_verbose(f"{ind_file_name} is previously added to the index and won't be downloaded.")
            continue
        tbd_indfileid2name[ind_file_id] = ind_file_name

    print_verbose(f'found {len(tbd_indfileid2name)} new indices files.')

    # ----- Download and store -----
    t0 = time.time()
    job_cnt = 0
    for ind_file_id, ind_file_name in tbd_indfileid2name.items():
        # Download the new indices
        gdu.download_file(ind_file_id, ind_file_name, params['indices_downloaded_path'])

        # Load the new indices
        with open(os.path.join(params['indices_downloaded_path'], ind_file_name), 'rb') as f:
            # noinspection PyTypeChecker
            new_indices = np.load(f)

        # Check if any exists in current indices
        print_verbose('checking for duplicates in the new indices ...')

        keep_mask = [(idx not in all_indices) for idx in new_indices]

        print_verbose('done!')

        if not np.any(keep_mask):
            job_cnt += 1
            continue

        print_verbose('number of new embedding vectors: %d' % np.sum(keep_mask))

        # Find the embeddings
        emb_file_name = ind_file_name.replace('indices', 'embeddings')
        emb_file_id = None
        for f_id in embfileid2name:
            if embfileid2name[f_id] == emb_file_name:
                emb_file_id = f_id
                break

        if emb_file_id is None:
            raise Exception('Cannot locate the embedding in gdrive.')

        # Download the new embeddings
        gdu.download_file(emb_file_id, emb_file_name, params['embeddings_downloaded_path'])

        # Load the new embeddings
        with open(os.path.join(params['embeddings_downloaded_path'], emb_file_name), 'rb') as f:
            # noinspection PyTypeChecker
            new_embeddings = np.load(f)

        # Self-destruct
        if params['self_destruct']:
            print_verbose(f'removing {emb_file_name} from the disk ...')

            os.remove(os.path.join(params['embeddings_downloaded_path'], emb_file_name))

            print_verbose('done!')

        # Add to the index
        print_verbose('adding to the index ...')
        faiss_index.add(new_embeddings[keep_mask])
        print_verbose('done!')

        # Save the index
        print_verbose('saving the index ...')
        faiss.write_index(faiss_index, params['faiss_index_path'])
        print_verbose('done!')

        # Update the indices
        all_indices = np.append(all_indices, new_indices[keep_mask])

        # Save the indices
        with open(params['indices_path'], 'wb') as f:
            # noinspection PyTypeChecker
            np.save(f, all_indices)

        # Log
        dt = time.time() - t0
        job_cnt += 1
        tot_job = len(tbd_indfileid2name)
        print_verbose('===== \n%d from %d done! (time since start: %g s, time per item: %g s, est. remaining: %g s)'
                      % (job_cnt, tot_job, dt, dt/job_cnt, (tot_job - job_cnt)*dt/job_cnt))

