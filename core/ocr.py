import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import configs
from utils import pytorch_utils as ptu


class TrOCR:
    def __init__(self, ver=configs.TrOCRConfig.DEFAULT_VERSION):
        self.ver = ver

        self.processor = TrOCRProcessor.from_pretrained(f'microsoft/{ver}')
        self.model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/{ver}')

        self.model.to(ptu.device)

    def recognize(self, image_boxes) -> str:
        pixel_values = self.processor(images=image_boxes, return_tensors='pt').pixel_values

        pixel_values.to(ptu.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=20)

            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            concatenated_text = ' '.join(generated_texts).lower()

            return concatenated_text
