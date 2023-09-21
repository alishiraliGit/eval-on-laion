import numpy as np
import torch
from transformers import BertTokenizer, BertModel

import configs
from utils import pytorch_utils as ptu


class Bert:
    def __init__(self, ver=configs.BertConfig.DEFAULT_VERSION):
        self.ver = ver

        self.tokenizer = BertTokenizer.from_pretrained(ver)
        self.model = BertModel.from_pretrained(ver)

        self.model.to(ptu.device)

    def text_embeds(self, texts) -> np.ndarray:
        tok_texts = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        tok_texts = tok_texts.to(ptu.device)

        with torch.no_grad():
            text_outputs = self.model(**tok_texts)

            text_embeds = text_outputs.pooler_output

            return ptu.to_numpy(text_embeds)
