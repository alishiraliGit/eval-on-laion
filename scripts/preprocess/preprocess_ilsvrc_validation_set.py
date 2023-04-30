import pickle
import sys
import os

import numpy as np
from scipy.io import loadmat
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs


if __name__ == '__main__':
    # ----- Settings -----
    settings = dict()

    # Path
    settings['devkit_path'] = os.path.join('ilsvrc2012', 'ILSVRC2012_devkit_t12')

    settings['dataframe_path'] = os.path.join('ilsvrc2012')
    settings['labels_path'] = os.path.join('ilsvrc2012', 'processed', 'labels')

    settings['safe'] = False

    # ----- Init. -----
    meta_path = os.path.join(settings['devkit_path'], 'data', 'meta.mat')
    val_ground_truth_path = os.path.join(settings['devkit_path'], 'data', 'ILSVRC2012_validation_ground_truth.txt')

    os.makedirs(settings['dataframe_path'], exist_ok=True)
    os.makedirs(settings['labels_path'], exist_ok=True)

    # Safety
    open_type = 'xb' if settings['safe'] else 'wb'

    # ----- Load -----
    meta_mat = loadmat(meta_path)['synsets'][:, 0]

    labels = pd.read_csv(val_ground_truth_path, header=None, names=['label'])

    # ----- Match labels with wnids -----
    label2wnid = {}
    for i_label in range(configs.ILSVRCConfigs.NUM_WNID):
        label2wnid[meta_mat[i_label][0][0, 0]] = meta_mat[i_label][1][0]

    # ----- Count -----
    wnid2count = {}

    for idx in labels.index:
        label = labels.loc[idx, 'label']
        wnid = label2wnid[label]

        if wnid not in wnid2count:
            wnid2count[wnid] = 0

        wnid2count[wnid] += 1

    counts = [count for _, count in wnid2count.items()]
    print(f'num. per wnid: {np.mean(counts)} +- {np.std(counts)}')

    # ----- Name2wnid -----
    imagename2wnid = {}

    for idx in labels.index:
        label = labels.loc[idx, 'label']
        wnid = label2wnid[label]

        image_name = 'ILSVRC2012_val_%08d.JPEG' % (idx + 1)

        imagename2wnid[image_name] = wnid

    with open(os.path.join(settings['labels_path'], 'imagename2wnid.pkl'), open_type) as f:
        pickle.dump(imagename2wnid, f)

    # ----- Create and save a dataframe -----
    df = pd.DataFrame(index=list(imagename2wnid.keys()))

    df.to_parquet(os.path.join(settings['dataframe_path'], 'ILSVRC2012_val.parquet'))
