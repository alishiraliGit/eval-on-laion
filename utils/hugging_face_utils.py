import torch

from utils import pytorch_utils as ptu


def get_label2wnids_map(model, id_lemmas_df, verbose=True):
    label2wnids = {}
    for label, lemmas in model.config.id2label.items():
        lemmas = lemmas.replace('_', ' ')
        wnids = id_lemmas_df.loc[id_lemmas_df['lemmas'] == lemmas, 'id'].values
        if len(wnids) == 0:
            if verbose:
                print('Cannot match label %d (%s).' % (label, lemmas))
            continue
        label2wnids[label] = wnids

    return label2wnids


def predict(processor, model, label2wnids, images):
    inputs = processor(images=images, return_tensors='pt')

    inputs.to(ptu.device)

    include_labels = list(label2wnids.keys())

    with torch.no_grad():
        outputs = model(**inputs)

        top5s = outputs.logits[:, include_labels].argsort(dim=1, descending=True)[:, :5]

        wnids_pr = []
        for top5 in top5s:
            wnids_pr_i = []
            for label_pr in top5:
                wnids_pr_i.extend(label2wnids[include_labels[label_pr.numpy().item()]])

            wnids_pr.append(wnids_pr_i)

        return wnids_pr
