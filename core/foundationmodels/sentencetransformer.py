import torch
import numpy as np
from sentence_transformers import SentenceTransformer

import configs
from utils import pytorch_utils as ptu


class SentTransformer:
    def __init__(self, ver=configs.SentTransformer.DEFAULT_VERSION):
        self.ver = ver

        self.model = SentenceTransformer(ver, device=ptu.device)

    def text_embeds(self, texts) -> np.ndarray:
        with torch.no_grad():
            text_embeds = self.model.encode(texts)

            return text_embeds
