import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification, \
    ResNetForImageClassification, \
    BeitImageProcessor, BeitForImageClassification, \
    ConvNextImageProcessor, ConvNextForImageClassification
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

import configs
from utils import pytorch_utils as ptu


class PredictorsPartition:
    IMAGENET_1K = 'imagenet-1k',
    IMAGENET_PT21k_FT1K = 'imagenet-pt21k-ft1k'
    IMAGENET_21K = 'imagenet-21k'


def select_imagenet_models(partitions):
    if not isinstance(partitions, list):
        partitions = [partitions]

    model_names = []
    processors = []
    models = []
    for partition in partitions:
        if partition == PredictorsPartition.IMAGENET_1K:
            model_names.extend(model_names_1k)
            processors.extend(processors_1k)
            models.extend(models_1k)
        elif partition == PredictorsPartition.IMAGENET_PT21k_FT1K:
            model_names.extend(model_names_pt21k_ft1k)
            processors.extend(processors_pt21k_ft1k)
            models.extend(models_pt21k_ft1k)
        elif partition == PredictorsPartition.IMAGENET_21K:
            model_names.extend(model_names_21k)
            processors.extend(processors_21k)
            models.extend(models_21k)
        else:
            raise Exception(f"Cannot find the model specified: {partition}")

    # Init.
    # TODO: Move to ptu.device
    processors = [processor() for processor in processors]
    models = [model() for model in models]

    return model_names, processors, models


###############
# 1k Models
###############
model_names_1k = ['ResNet-50-1k', 'ConvNeXT-1k']

processors_1k = {
    'ResNet-50-1k': lambda: ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224'),  # Works, w/o warning
    'ConvNeXT-1k': lambda: ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224')
}

models_1k = {
    'ResNet-50-1k': lambda: ResNetForImageClassification.from_pretrained('microsoft/resnet-50'),
    'ConvNeXT-1k': lambda: ConvNextForImageClassification.from_pretrained('facebook/convnext-base-224')
}


###############
# PT21k FT1k Models
###############
model_names_pt21k_ft1k = ['ConvNeXT-21k-1k', 'ViT-21k-1k', 'BEiT-21k-1k']

processors_pt21k_ft1k = {
    'ConvNeXT-21k-1k': lambda: ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224-22k-1k'),
    'ViT-21k-1k': lambda: ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'),
    'BEiT-21k-1k': lambda: BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
}

models_pt21k_ft1k = {
    'ConvNeXT-21k-1k': lambda: ConvNextForImageClassification.from_pretrained('facebook/convnext-base-224-22k-1k'),
    'ViT-21k-1k': lambda: ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
    'BEiT-21k-1k': lambda: BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
}


###############
# 21k Models
###############
model_names_21k = ['ConvNeXT-21k', 'BEiT-21k']

processors_21k = {
    'ConvNeXT-21k': lambda: ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224-22k'),
    'BEiT-21k': lambda: BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
}

models_21k = {
    'ConvNeXT-21k': lambda: ConvNextForImageClassification.from_pretrained('facebook/convnext-base-224-22k'),
    'BEiT-21k': lambda: BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
}


###############
# CLIP
###############
class CLIP:
    def __init__(self, ver=configs.CLIPConfig.DEFAULT_VERSION):
        self.processor = CLIPProcessor.from_pretrained(f'openai/{ver}')
        self.model = CLIPModel.from_pretrained(f'openai/{ver}')

        # Only for obtaining text embeddings
        self.text_processor = CLIPTokenizer.from_pretrained(f'openai/{ver}')

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
