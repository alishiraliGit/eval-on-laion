1. Download and unzip the three versions of ImageNet-V2 here.
2. Run [preprocess_imagenet_v2.py](../scripts/preprocess/preprocess_imagenet_v2.py) for each version separately
by specifying each version's folder of images in `--images_path`. 
This should create three dataframes under the current folder 
with their labels under [imagenetv2/processed/labels](processed/labels).