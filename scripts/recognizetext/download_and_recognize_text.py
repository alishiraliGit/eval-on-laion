import sys
import os
import multiprocessing
import time
import argparse
import pandas as pd
from tqdm.auto import tqdm
import traceback

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.text_detectors import EAST
from core.ocr import TrOCR


###############
# Image download
###############

from scripts.predict.download_and_predict import download_image_wrapper, unpaused, setup


###############
# Text detection and recognition
###############

text_detector: EAST = None
text_recognizer: TrOCR = None


def load_models():
    global text_detector, text_recognizer

    if text_detector is None:
        text_detector = EAST()

    if text_recognizer is None:
        text_recognizer = TrOCR()

    return text_detector, text_recognizer


def init_worker(pars):
    global text_detector, text_recognizer

    worker_id = multiprocessing.current_process().name

    logu.verbose = pars['verbose']

    # Init
    print_verbose(f'initializing worker {worker_id} ...')

    ptu.init_gpu(use_gpu=not pars['no_gpu'], gpu_id=pars['gpu_id'])

    print_verbose('done!\n')

    # Load the models
    print_verbose(f'loading text detector and recognizer in worker {worker_id} ...')

    load_models()

    print_verbose('done!\n')


def detect_and_recognize(args):
    global text_detector, text_recognizer

    idx_img_list = args

    # Load the images and recognize
    empty_inds = []
    txt_inds = []
    txts = []

    err_inds = []
    errs = []
    for idx, img_content in idx_img_list:
        try:
            _, img_boxes = text_detector.detect_and_select_boxes(img_content, draw_box=False)

            if len(img_boxes) == 0:
                empty_inds.append(idx)
                continue

            txt = text_recognizer.recognize(img_boxes)

            txt_inds.append(idx)
            txts.append(txt)

        except Exception as e:
            err_inds.append(idx)
            errs.append({
                'cause': 'In detection or recognizing texts of a batch of images an error occurred.',
                'error': e}
            )

    return empty_inds, txt_inds, txts, err_inds, errs


def dummy_func(_args):
    return None


###############
# Progress monitor
###############

def num_ready_results(results):
    return sum([r.ready() for r in results])


def update_recognition_pb(rec_pb, rec_results):
    rec_pb.total = len(rec_results)
    rec_pb.n = num_ready_results(rec_results)
    rec_pb.refresh()


###############
# Handle results
###############

def wait_for_recognition(rec_pb, rec_results, margin=10):
    update_recognition_pb(rec_pb, rec_results)

    while num_ready_results(rec_results) < len(rec_results) - margin:
        update_recognition_pb(rec_pb, rec_results)
        time.sleep(0.1)


def collect_recognition_results(rec_results):
    empty_indices = []
    text_indices = []
    texts = []
    rec_error_indices = []

    for i_rec, res in enumerate(tqdm(rec_results, desc='collecting the results', leave=True)):
        try:
            empty_indices_i, text_indices_i, texts_i, rec_error_indices_i, rec_errors_i = res.get(timeout=10)
        except multiprocessing.TimeoutError as e:
            print_verbose('recognition pool went timeout for one job.')
            print(str(e))
            continue

        # Append errors
        for err in rec_errors_i:
            errors.append('\n' + err['cause'])
            errors.append(str(err['error']))

        empty_indices.extend(empty_indices_i)
        text_indices.extend(text_indices_i)
        texts.extend(texts_i)
        rec_error_indices.extend(rec_error_indices_i)

    time.sleep(1)

    # ----- Update df -----
    print_verbose('\n\nupdating the df ...')

    df_all.loc[empty_indices, rec_text_col] = 'NO TEXT'

    df_all.loc[text_indices, rec_text_col] = texts

    df_all.loc[rec_error_indices, rec_text_col] = 'RECOGNITION ERROR'

    print_verbose('done!\n')


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'text_recognition'))

    # Subset
    parser.add_argument('--from_iloc', type=int, default=0)
    parser.add_argument('--to_iloc', type=int, default=-1)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=2)
    parser.add_argument('--n_process_recognition', type=int, default=6)
    parser.add_argument('--recognition_max_todo', type=int, default=1000)  # 1000 roughly corresponds to 4GB in RAM

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--change_ver_freq', type=int, default=10)

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Prefix
    prefix = params['prefix']

    # Recognized text column
    rec_text_col = 'recognized_text'

    # Download EAST if required
    EAST.download()

    # Logging
    file_ver = time.strftime('%d-%m-%Y_%H-%M-%S')

    print_verbose('done!\n')

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df_all = pd.read_parquet(subset_file_path)

    # Subset
    if params['to_iloc'] > 0:
        df = df_all.iloc[params['from_iloc']: params['to_iloc']]
    else:
        df = df_all.iloc[params['from_iloc']:]

    if rec_text_col in df:
        df = df[df[rec_text_col].isnull()]

    print_verbose('done!\n')

    # ----- Init. parallel download and predict -----
    event = multiprocessing.Event()
    pool_download = multiprocessing.Pool(params['n_process_download'], setup, (event,))
    pool_recognition = multiprocessing.Pool(params['n_process_recognition'], initializer=init_worker, initargs=(params,))

    # Init pool_recognition
    pool_recognition.map(dummy_func, [1] * params['n_process_recognition'])

    # ----- Start download -----
    download_results = pool_download.imap(download_image_wrapper, df.iterrows())
    event.set()

    # ----- Download and send for recognition -----
    download_ready_results = []
    recognition_results = []

    download_error_indices = []
    errors = []

    recognition_pb = tqdm(desc='recognizing', leave=True)
    save_cnt = 0
    for down_res in tqdm(download_results, desc='(mostly) downloading', total=len(df), leave=True):
        # Catch the errors in downloading
        index, image_content, error = down_res
        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
            download_error_indices.append(index)
            continue

        # Append the downloaded image to the batch
        download_ready_results.append([index, image_content])
        if len(download_ready_results) < configs.EASTConfig.BATCH_SIZE:
            continue

        # Send the batch for recognition
        rec_res = pool_recognition.apply_async(
            detect_and_recognize,
            [download_ready_results]
        )
        recognition_results.append(rec_res)

        # Empty current batch
        download_ready_results = []

        # Monitor recognition progress
        update_recognition_pb(recognition_pb, recognition_results)

        # Wait for recognition
        if len(recognition_results) - num_ready_results(recognition_results) > params['recognition_max_todo']:
            event.clear()

            # Mark download errors
            df_all.loc[download_error_indices, rec_text_col] = 'DOWNLOAD ERROR'
            download_error_indices = []

            # Wait for recognition
            wait_for_recognition(recognition_pb, recognition_results)

            # Collect the results and update the df
            collect_recognition_results(recognition_results)
            recognition_results = []

            # Save
            print_verbose('saving ....')

            save_cnt += 1

            if save_cnt % params['change_ver_freq'] == 0:
                file_ver = time.strftime('%d-%m-%Y_%H-%M-%S')

            err_file_path = subset_file_path.replace('snappy.parquet', f'textrecognition_errors_{file_ver}.txt')
            with open(err_file_path, 'w') as f:
                f.write('\n'.join(errors))

            save_subset_file_path = subset_file_path.replace('.snappy', f'_{file_ver}.snappy')
            df_all.to_parquet(save_subset_file_path, index=True)

            print_verbose('done!\n')

            time.sleep(1)

        event.set()

    # Mark download errors
    df_all.loc[download_error_indices, rec_text_col] = 'DOWNLOAD ERROR'

    # Send the remaining for recognition
    rec_res = pool_recognition.apply_async(
        detect_and_recognize,
        [download_ready_results]
    )
    recognition_results.append(rec_res)

    # ----- Waiting for recognition -----
    wait_for_recognition(recognition_pb, recognition_results)

    # ----- Collect the results ------
    collect_recognition_results(recognition_results)
    recognition_results = []

    # ----- Close progress bars and processes -----
    recognition_pb.close()

    pool_download.close()
    pool_download.join()

    pool_recognition.close()
    pool_recognition.join()

    time.sleep(3)

    # ----- Save ------
    print_verbose('saving ....')

    err_file_path = subset_file_path.replace('parquet', f'textrecognition_errors_{file_ver}.txt')
    with open(err_file_path, 'w') as f:
        f.write('\n'.join(errors))

    save_subset_file_path = subset_file_path.replace('.snappy', f'_{file_ver}.snappy')
    df_all.to_parquet(save_subset_file_path, index=True)

    print_verbose('done!\n')
