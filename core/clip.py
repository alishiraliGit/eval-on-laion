import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor

import configs
from utils import pytorch_utils as ptu


class CLIP:
    def __init__(self, ver=configs.CLIPConfig.DEFAULT_VERSION):
        self.processor = CLIPProcessor.from_pretrained(f'openai/{ver}')
        self.model = CLIPModel.from_pretrained(f'openai/{ver}')

        # Only for obtaining text embeddings
        self.text_processor = CLIPTokenizer.from_pretrained(f'openai/{ver}')

        # Only for obtaining image embeddings
        self.image_processor = CLIPImageProcessor.from_pretrained(f'openai/{ver}')

        self.model.to(ptu.device)

    def similarities(self, texts, images) -> np.ndarray:
        inputs = self.processor(text=texts, images=images, return_tensors='pt', padding=True, truncation=True)

        inputs.to(ptu.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            sims = torch.diag(outputs.logits_per_image)

            # Divide by 100: Based on my observations, this number will be cosine similarity.
            return ptu.to_numpy(sims)/100

    def text_embeds(self, texts) -> np.ndarray:
        inputs = self.text_processor(texts, return_tensors='pt', padding=True, truncation=True)

        inputs.to(ptu.device)

        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)

            embeds = self.model.text_projection(text_outputs.pooler_output)

            return ptu.to_numpy(embeds)

    def image_embeds(self, images) -> np.ndarray:
        inputs = self.image_processor(images, return_tensors='pt')

        inputs.to(ptu.device)

        with torch.no_grad():
            image_outputs = self.model.vision_model(**inputs)

            embeds = self.model.visual_projection(image_outputs.pooler_output)

            return ptu.to_numpy(embeds)
