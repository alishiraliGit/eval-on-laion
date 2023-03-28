import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

import configs


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


class CLIP:
    def __init__(self, ver=configs.CLIPConfig.DEFAULT_VERSION):
        self.processor = CLIPProcessor.from_pretrained(f'openai/{ver}')
        self.model = CLIPModel.from_pretrained(f'openai/{ver}')

    def similarities(self, texts, images, trunc=True) -> np.ndarray:
        inputs = self.processor(text=texts, images=images, return_tensors='pt', padding=True)

        if trunc:
            inputs['input_ids'] = inputs['input_ids'][:, :configs.CLIPConfig.MAX_SEQ_LENGTH]
            inputs['attention_mask'] = inputs['attention_mask'][:, :configs.CLIPConfig.MAX_SEQ_LENGTH]

        outputs = self.model(**inputs)

        sims = torch.diag(outputs.logits_per_image)
        # Divide by 100: Based on my observations, this number will be cosine similarity.
        return sims.detach().numpy()/100