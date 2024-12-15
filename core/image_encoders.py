import numpy as np
import torch
from transformers import ViTImageProcessor, ViTModel, BeitImageProcessor, BeitModel, \
    ConvNextImageProcessor, ConvNextModel

import configs
from core.foundationmodels.clip import CLIP
from utils import pytorch_utils as ptu

img_encoder_ver = None
img_encoder = None
img_encoder_batch_size = None


def select_image_encoder(ver:  str):
    global img_encoder_ver, img_encoder, img_encoder_batch_size

    if ver == img_encoder_ver:
        return img_encoder, img_encoder_batch_size

    if ver.startswith('clip'):
        clip = CLIP(ver=ver)

        img_encoder = clip.image_embeds
        img_encoder_ver = ver
        img_encoder_batch_size = configs.CLIPConfig.BATCH_SIZE

    elif ver.startswith('vit'):
        wrapper = ViTWrapper(ver)

        img_encoder = wrapper.image_embeds
        img_encoder_ver = ver
        img_encoder_batch_size = configs.CLIPConfig.BATCH_SIZE

    elif ver.startswith('beit'):
        wrapper = BEiTWrapper(ver)

        img_encoder = wrapper.image_embeds
        img_encoder_ver = ver
        img_encoder_batch_size = configs.CLIPConfig.BATCH_SIZE

    elif ver.startswith('convnext'):
        wrapper = ConvNeXTWrapper(ver)

        img_encoder = wrapper.image_embeds
        img_encoder_ver = ver
        img_encoder_batch_size = configs.CLIPConfig.BATCH_SIZE

    else:
        raise NotImplementedError

    return img_encoder, img_encoder_batch_size


class ViTWrapper:
    def __init__(self, ver):
        self.ver = ver

        self.processor = ViTImageProcessor.from_pretrained(f'google/{ver}')
        self.model = ViTModel.from_pretrained(f'google/{ver}')

        self.model.to(ptu.device)

    def image_embeds(self, images) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors='pt')

        inputs = inputs.to(ptu.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeds = outputs.pooler_output

        return ptu.to_numpy(embeds)


class BEiTWrapper:
    def __init__(self, ver):
        self.ver = ver

        self.processor = BeitImageProcessor.from_pretrained(f'microsoft/{ver}')
        self.model = BeitModel.from_pretrained(f'microsoft/{ver}')

        self.model.to(ptu.device)

    def image_embeds(self, images) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors='pt')

        inputs = inputs.to(ptu.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeds = outputs.pooler_output

        return ptu.to_numpy(embeds)


class ConvNeXTWrapper:
    def __init__(self, ver):
        self.ver = ver

        self.processor = ConvNextImageProcessor.from_pretrained(f'facebook/{ver}')
        self.model = ConvNextModel.from_pretrained(f'facebook/{ver}')

        self.model.to(ptu.device)

    def image_embeds(self, images) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors='pt')

        inputs = inputs.to(ptu.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeds = outputs.pooler_output

        return ptu.to_numpy(embeds)
