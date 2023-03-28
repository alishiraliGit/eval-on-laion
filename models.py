from transformers import ViTImageProcessor, ViTForImageClassification, \
    AutoImageProcessor, ResNetForImageClassification, \
    BeitImageProcessor, BeitForImageClassification, \
    ConvNextImageProcessor, ConvNextForImageClassification

# ----- 1k models -----
model_names_1k = ['ResNet-50', 'ConvNeXT', 'ViT', 'BEiT']

processors_1k = {
    'ResNet-50': ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224'),
    'ConvNeXT': ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224'),
    'ViT': ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'),
    'BEiT': BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
}

models_1k = {
    'ResNet-50': ResNetForImageClassification.from_pretrained('microsoft/resnet-50'),
    'ConvNeXT': ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224'),
    'ViT': ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
    'BEiT': BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
}

# ----- 21k models -----
model_names = ['ConvNeXT-21k', 'BEiT-21k']

processors = {
    'ConvNeXT-21k': ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224-22k'),
    'BEiT-21k': BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
}

models = {
    'ConvNeXT-21k': ConvNextForImageClassification.from_pretrained('facebook/convnext-base-224-22k'),
    'BEiT-21k': BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
}

