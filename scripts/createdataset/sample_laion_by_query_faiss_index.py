import sys
import os
import argparse
import pickle
# FAISS will crash if you don't import it here!
import faiss

from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from core.clip import CLIP
from core.faiss_index import FaissIndex
from core.queries import QueryType, select_queries
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils.ilsvrc_utils import load_lemmas_and_wnids


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--ilsvrc_synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))

    parser.add_argument('--indices_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices', 'all_indices.npy'))

    parser.add_argument('--faiss_index_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'faiss_index', 'knn.index'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

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

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Load synsets -----
    id_lemmas_df = load_lemmas_and_wnids(params['ilsvrc_synsets_path'])
    wnids = id_lemmas_df['id'].tolist()

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load the text index -----
    faiss_index = FaissIndex.load(params['faiss_index_path'], params['indices_path'])

    # ----- Specify queries -----
    query_types = [
        QueryType.NAME,
        QueryType.NAME_DEF,
        QueryType.LEMMAS,
        QueryType.A_PHOTO_OF_NAME,
        QueryType.A_PHOTO_OF_NAME_DEF,
        QueryType.A_PHOTO_OF_LEMMAS
    ]

    query_funcs = select_queries(query_types)

    # ----- Loop over wnids and query the index -----
    for i_q, query_func in tqdm(enumerate(query_funcs), desc='query the index'):
        wnid2laionindices = {}
        wnid2cossims = {}

        # Query all wnids
        for wnid in tqdm(wnids, desc=f'query: {query_types[i_q]}'):
            # Get the query text
            q = query_func(wnid)

            # Get CLIP embedding
            embed = clip.text_embeds([q])
            embed_norm = normalize(embed, norm='l2', axis=1)

            # Search the index
            indices, cos_sims = faiss_index.search(embed_norm, configs.LAIONSamplingConfig.UNIFORM_SAMPLES)

            # Add
            wnid2laionindices[wnid] = indices
            wnid2cossims[wnid] = cos_sims

        # Save
        file_name = f'wnid2laionindices(query_{query_types[i_q]}).pkl'
        with open(os.path.join(params['save_path'], file_name), open_type) as f:
            pickle.dump(wnid2laionindices, f)

        sim_file_name = f'wnid2cossims(query_{query_types[i_q]}).pkl'
        with open(os.path.join(params['save_path'], sim_file_name), open_type) as f:
            pickle.dump(wnid2cossims, f)
