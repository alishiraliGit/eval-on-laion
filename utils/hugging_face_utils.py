import torch

from utils import pytorch_utils as ptu
from utils.logging_utils import print_verbose


def get_label2wnid_map(model, id_lemmas_df):
    label2wnid = {}
    for label, lemmas in model.config.id2label.items():
        lemmas = lemmas.replace('_', ' ')
        wnids = id_lemmas_df.loc[id_lemmas_df['lemmas'] == lemmas, 'id'].values
        assert len(wnids) <= 1
        if len(wnids) == 0:
            continue
        label2wnid[label] = wnids[0]

    print_verbose(f'\tmatched {len(label2wnid)} labels to a wnid.')

    return label2wnid


def predict(processor, model, label2wnid, images, k=5):
    inputs = processor(images=images, return_tensors='pt')

    inputs.to(ptu.device)

    include_labels = list(label2wnid.keys())

    with torch.no_grad():
        outputs = model(**inputs)

        topks = outputs.logits[:, include_labels].argsort(dim=1, descending=True)[:, :k]

        topks = ptu.to_numpy(topks)

        wnids_pr = []
        for topk in topks:
            wnids_pr_i = [label2wnid[include_labels[label_pr.item()]] for label_pr in topk]

            wnids_pr.append(wnids_pr_i)

        return wnids_pr
