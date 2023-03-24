from transformers import ViTImageProcessor, ViTForImageClassification, \
    AutoImageProcessor, ResNetForImageClassification, \
    BeitImageProcessor, BeitForImageClassification, \
    ConvNextImageProcessor, ConvNextForImageClassification

model_names = ['ResNet-50', 'ConvNeXT', 'ViT', 'BEiT']

processors = {
    'ResNet-50': ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224'),
    'ConvNeXT': ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224'),
    'ViT': ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'),
    'BEiT': BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
}

models = {
    'ResNet-50': ResNetForImageClassification.from_pretrained('microsoft/resnet-50'),
    'ConvNeXT': ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224'),
    'ViT': ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
    'BEiT': BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
}