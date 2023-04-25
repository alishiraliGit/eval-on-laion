from transformers import ViTImageProcessor, ViTForImageClassification, \
    ResNetForImageClassification, \
    BeitImageProcessor, BeitForImageClassification, \
    ConvNextImageProcessor, ConvNextForImageClassification
from utils import pytorch_utils as ptu


class ILSVRCPredictorType:
    IMAGENET_1K = 'imagenet-1k'
    IMAGENET_PT21k_FT1K = 'imagenet-pt21k-ft1k'
    IMAGENET_21K = 'imagenet-21k'
    IMAGENET_VIT = 'imagenet-vit'


def select_ilsvrc_predictors(types):
    if not isinstance(types, list):
        types = [types]

    model_names = []
    processors = {}
    models = {}
    for t in types:
        if t == ILSVRCPredictorType.IMAGENET_1K:
            model_names.extend(model_names_1k)
            processors.update(processors_1k)
            models.update(models_1k)
        elif t == ILSVRCPredictorType.IMAGENET_PT21k_FT1K:
            model_names.extend(model_names_pt21k_ft1k)
            processors.update(processors_pt21k_ft1k)
            models.update(models_pt21k_ft1k)
        elif t == ILSVRCPredictorType.IMAGENET_21K:
            model_names.extend(model_names_21k)
            processors.update(processors_21k)
            models.update(models_21k)
        elif t == ILSVRCPredictorType.IMAGENET_VIT:
            model_names.extend(model_names_vit)
            processors.update(processors_vit)
            models.update(models_vit)
        else:
            raise Exception(f'Cannot find the model specified: {t}')

    # Init.
    for model_name in model_names:
        processors[model_name] = processors[model_name]()
        models[model_name] = models[model_name]().to(ptu.device)

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
# Other ViT models
###############
model_names_vit = [
    'vit-base-patch16-224',
    'vit-base-patch16-384',
    'vit-base-patch32-384',
    'vit-large-patch16-224',
    'vit-large-patch16-384',
    'vit-large-patch32-384'
]

processors_vit = {
    lambda: ViTImageProcessor.from_pretrained(f'google/{mdl_name}') for mdl_name in model_names_vit
}

models_vit = {
    lambda: ViTForImageClassification.from_pretrained(f'google/{mdl_name}') for mdl_name in model_names_vit
}
