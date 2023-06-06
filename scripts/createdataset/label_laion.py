import sys
import os
import argparse
import pickle
from tqdm import tqdm
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu


def label(args):
    idx, row, lem2wnid = args
    try:
        txt = laionu.transform_text(row[configs.LAIONConfig.TEXT_COL])

        lems = []
        ws = set()
        for lem, w in lem2wnid.items():
            if laionu.transform_lemma(lem) in txt:
                lems.append(lem)
                ws.add(w)

        return idx, lems, ws, None
    except Exception as e:
        return idx, None, None, {'cause': f'In labeling {idx} an error occurred.', 'error': e}


def df_wrapper(df, lem2wnid):
    for idx, row in df.iterrows():
        yield idx, row, lem2wnid


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--lemma2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed',
                                             'lemma2wnid(unique_in_ilsvrc_ignored_empty_wnids).pkl'))

    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_part', type=int)

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

    # Multiprocessing
    parser.add_argument('--n_process', type=int, default=8)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false', help='If set, overwriting will be allowed.')

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

    # ----- Load lemma2wnid -----
    print_verbose('loading lemma2wnid ...')

    with open(params['lemma2wnid_path'], 'rb') as f:
        lemma2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load LAION part -----
    data_frame = laionu.load_data_part(params['laion_path'], params['laion_part'], params['self_destruct'])

    # ----- Start a pool -----
    pool = multiprocessing.Pool(params['n_process'])

    results = pool.imap(label, df_wrapper(data_frame, lemma2wnid))

    # ----- Collect results -----
    lemma2laionindices = {}
    wnid2laionindices = {}
    errors = []

    for res in tqdm(results, desc=f'processing laion part {params["laion_part"]}', total=len(data_frame), leave=True):
        # Catch the errors
        index, lemmas, wnids, error = res
        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
            continue

        # Add to maps
        for lemma in lemmas:
            if lemma not in lemma2laionindices:
                lemma2laionindices[lemma] = []
            lemma2laionindices[lemma].append(index)

        for wnid in wnids:
            if wnid not in wnid2laionindices:
                wnid2laionindices[wnid] = []
            wnid2laionindices[wnid].append(index)

    # ----- Saving -----
    print(f'saving labels of laion part {params["laion_part"]} ...')

    wnid_map_file_name = f'wnid2laionindices(substring_matched_part{params["laion_part"]}).pkl'
    lemma_map_file_name = f'lemma2laionindices(substring_matched_part{params["laion_part"]}).pkl'

    with open(os.path.join(params['save_path'], wnid_map_file_name), open_type) as f:
        pickle.dump(wnid2laionindices, f)

    with open(os.path.join(params['save_path'], lemma_map_file_name), open_type) as f:
        pickle.dump(lemma2laionindices, f)

    print('done!\n')
