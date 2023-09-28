import numpy as np
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from utils import utils


top_k_col = lambda k, mdl: f'top_{k}_is_correct_{mdl}'
alpha = 0.05
z = 1.96


def calc_recall_per_y(df, y2indices, mdl_names, k, drop_if_true_cols=None, verbose=True):
    global top_k_col, alpha, z

    if drop_if_true_cols is not None:
        for col in drop_if_true_cols:
            df = df[~df[col]]

    mdl2y2recall = {mdl_name: {} for mdl_name in mdl_names}
    mdl2y2recallse = {mdl_name: {} for mdl_name in mdl_names}  # standard err
    mdl2ys = {mdl_name: [] for mdl_name in mdl_names}

    cols = [top_k_col(k, mdl_name) for mdl_name in mdl_names]

    df_index = list(df.index)
    for y, indices in tqdm(y2indices.items(), disable=not verbose):
        indices = utils.intersect_lists(indices, df_index)

        vals = df.loc[indices, cols].to_numpy()

        mask = vals != None

        ns = np.sum(mask, axis=0)

        for i_mdl, mdl_name in enumerate(mdl_names):
            n = ns[i_mdl]

            count = np.sum(vals[mask[:, i_mdl], i_mdl], axis=0)

            if n == 0:
                continue

            mdl2ys[mdl_name].append(y)

            recall = count / n

            mdl2y2recall[mdl_name][y] = recall
            ci_l, ci_u = proportion_confint(count, n, method='wilson', alpha=alpha)
            mdl2y2recallse[mdl_name][y] = (ci_u - ci_l) / 2 / z

    return mdl2y2recall, mdl2y2recallse, mdl2ys


def combine_recalls(means, ses, w):
    nan_filt = ~np.isnan(means)
    means = means[nan_filt]
    ses = ses[nan_filt]
    w = w[nan_filt]

    mean = np.sum(means * w) / np.sum(w)
    se = np.sqrt(np.sum((ses * w) ** 2)) / np.sum(w)

    return mean, se


def calc_equi_acc(mdl_names, mdl2y2recall, mdl2y2recallse):
    mdl2acc = {}
    mdl2ny = {}
    for mdl_name in mdl_names:
        recalls = []
        recallses = []
        for y, recall in mdl2y2recall[mdl_name].items():
            recalls.append(recall)
            recallses.append(mdl2y2recallse[mdl_name][y])

        recalls = np.array(recalls)
        recallses = np.array(recallses)

        acc, acc_se = combine_recalls(recalls, recallses, np.ones(recalls.shape))

        mdl2acc[mdl_name] = {
            'acc': acc,
            'acc_se': acc_se,
        }

        mdl2ny[mdl_name] = np.sum(~np.isnan(recalls))

    return mdl2acc, mdl2ny
