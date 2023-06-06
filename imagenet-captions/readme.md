1. Download `imagenet_captions.zip` from https://github.com/mlfoundations/imagenet-captions 
and unzip it to obtain the `imagenet_captions.json`.
2. Run [preprocess_imagenet_captions.py](../scripts/preprocess/preprocess_imagenet_captions.py). 
This will create a dataframe at this folder and a dictionary containing WNID of each image at
[imagenet-captions/processed/labels](processed/labels).
3. Download ILSVRC2012 training images from https://www.image-net.org/download.php 
and place them under [ilsvrc2012/ILSVRC2012_img_train](../ilsvrc2012/ILSVRC2012_img_train). This is a temporary place.
4. Run [drop_imagenet_examples_wo_caption.py](../scripts/preprocess/drop_imagenet_examples_wo_caption.py).
This will copy ILSVRC training images which are present in ImageNet-Captions to 
[ilsvrc2012/ILSVRC2012_img_train_selected](../ilsvrc2012/ILSVRC2012_img_train_selected).
You can now remove the images left in [ilsvrc2012/ILSVRC2012_img_train](../ilsvrc2012/ILSVRC2012_img_train).