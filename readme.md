This is the accompanying code of "What Makes ImageNet Look Unlike LAION."
(<a href="https://arxiv.org/abs/2306.15769" target="_blank">link to the paper</a>)

This page explains how to
1. Create LAIONet (Section 2 of the paper), 
2. Evaluate the accuracy of various models on LAIONet and compare it to their accuracy on ImageNet ILSVRC2012 (Section 3.1), 
3. Analyze and contrast intra-class similarity of LAIONet, ImageNet, and ImageNet-V2 (Sections 3.2 and 4.1) 
4. Reproduce the results of experiments in Sections 4.2 and 4.3 of the paper.

# Setup
We recommend cloning the project, creating a virtual environment,
and installing the required packages from [requirements.txt](requirements.txt):
```shell
python -m venv laionvenv
source laionvenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Create LAIONet
Creating LAIONet involves three steps: 
1. Substring match LAION texts with the lemmas of ILSVRC 1000 synsets. 
2. Using CLIP, calculate the similarity of LAION texts to the textual representations of their matched synsets.
3. Select instances with high similarity to their associated synsets to create LAIONet.

In the following, we explain the commands and execution details for each step.

## 1. Find lemmas in LAION texts
Run script [label_laion.py](scripts/createdataset/label_laion.py) to substring match LAION texts with lemmas in 1000 
ILSVRC synsets. The script requires a dictionary, mapping lemmas to their WNIDs. 
We recommend using the [default option](ilsvrc2012/processed/lemma2wnid(unique_in_ilsvrc_ignored_empty_wnids).pkl) 
which only includes lemmas that are unique across synsets. 
The script should be run for each of the 32 parts of LAION-400M.
The parameter `--laion_part` determines the part, taking a value from 0 to 31. 
It also allows parallel processing by setting `--n_process`. 
Make sure to set `--self_destruct` if you want to remove the downloaded LAION part after the process is done. 
Use script [label_laion_parallel.py](scripts/createdataset/label_laion_parallel.py) 
to run on multiple LAION parts in parallel. 
Look at the scripts for further arguments and help.

Using default parameters, the previous scripts save a dictionary per each part 
in folder [laion400m/processed/ilsvrc_labels](laion400m/processed/ilsvrc_labels). 
Each dictionary maps WNID to LAION indices for that part.  
Use the following command to extract a dataset from all LAION instances that are matched to at least one lemma:
```shell
python scripts/createdataset/subset_laion.py \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--method substring_matched \
--self_destruct
```
The `--method` argument is only for consistent naming throughout the process and `--self_destruct` ensures 
LAION original files will be removed from the disk by process completion.
This command will create a large dataset in `.parquet` format under folder [laion400m](laion400m) 
occupying ~4 GB of the disk. 

## 2. Calculate LAION text to synset text similarity
The first step to calculate textual similarity of a LAION instance and a synset is to create a textual representation
for the synset. Generally, we call such a representation a query.
Following the paper, we recommend concatenating the name of the synset and its short definition. 
This corresponds to `QueryType.NAME_DEF`. Look at [queries.py](core/queries.py) for a list of possible queries.
Run the following command to calculate LAION text to synset text (query) similarity 
for the dataset we created in the previous step:
```shell
python scripts/calcsimilarity/calc_and_store_clip_text_to_query_similarities.py \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--method substring_matched \
--query_type name_def \
--query_key wnid \
--gpu_id 0
```
The above command allows specifying `--gpu_id` or `no_gpu`. Look at the script for further arguments and help.
We recommend using a single GPU for this task to accelerate calculation of CLIP embeddings. 
Upon completion, a new column `text_to_name_def_wnid_similarity` will be added to the dataset. 
We drop LAION instances which have lemmas from multiple synsets at this step.

## 3. Filter out LAION instances with low similarity to their synsets
Run the following command to filter out LAION instances with low textual similarity to their associated synsets:
```shell
python scripts/createdataset/sample_laion_from_most_similars.py \
--similarity_th 0.82 \
--remove_nsfw \
--method substring_matched_filtered
```
We followed the paper and set the minimum required similarity to 0.82. 
Follow notebook [choose_text_query_similarity_threshold.ipynb](notebooks/choose_text_query_similarity_threshold.ipynb)
to see this is a conservative choice and try out other values.
Setting `--remove_nsfw` removes the instances labeled to be not safe for work. 
Using default parameters, the script creates a new dataset, to be called LAIONet, under [laion400m](laion400m) folder. 
It also saves a new dictionary in [laion400m/processed/ilsvrc_labels](laion400m/processed/ilsvrc_labels) 
which maps WNID to the indices of LAIONet.


# Find and compare accuracy
## Find accuracy on LAIONet
Like LAION, LAIONet created in the previous section solely includes the URLs of the images 
rather than the images themselves. Consequently, downloading the images is necessary to evaluate any model.
Currently, our codebase supports the evaluation of any model compatible with HuggingFace classifier in an efficient way.
We particularly used the models listed in [ilsvrc_predictors.py](core/ilsvrc_predictors.py). 
But LAIONet is self-contained and researchers can use their own scripts to evaluate other models based on LAIONet.

To download the LAIONet images on the fly and predict their labels using selected models, use the following command:
```shell
python scripts/predict/download_and_predict.py \
--method substring_matched_filtered \
--predictors selected \
--n_process_download 16 \
--n_process_predict 1 \
--pred_max_todo 500 \
--gpu_id 0
```
Beyond selected models, the argument `--predictors` allows for
any `ILSVRCPredictorType` from [ilsvrc_predictors.py](core/ilsvrc_predictors.py) 
or can be set to `all` to include all models. 
The remaining parameters determine the degree of multiprocessing. 
In this case, we have utilized 16 processes for image downloading and one process for predicting their labels.
It is worth noting that parallelization greatly enhances the download process. 
Setting number of processes beyond the number of your processor cores is not recommended.
Additionally, we highly recommend using a single GPU to expedite model evaluation. 
Sometimes downloading is faster than prediction.
The parameter `--pred_most_todo` determines the maximum number of batches downloaded but not predicted. 
After this threshold, the program pauses the download and wait for the prediction to catch up. 
To change the batch size look at `ILSVRCPredictorsConfig` in [configs.py](configs.py).

The above command saves one `.csv` file for each model in 
[laion400m/processed/ilsvrc_predictions](laion400m/processed/ilsvrc_predictions). 
The file comes with the name of the model and date and contains predicted WNIDs.
To evaluate these predictions, use 
```shell
python scripts/postprocess/evaluate_predictions_on_laion_subset.py \
--labels_file_name "wnid2laionindices(substring_matched).pkl" \
--method substring_matched_filtered \
--predictors selected 
```
This command will add one binary column for each model in `--predictors` to LAIONet. 
It also allows selecting specific version of predictions by setting `--predictions_ver`. 
An average of these columns give accuracy of each predictor. 
Look at the script for further options and guidance.

## Find accuracy on ILSVRC2012 validation set
Download the images of ILSVRC2012 validation set from https://www.image-net.org/download.php and put them 
under folder [ilsvrc2012/ILSVRC2012_img_val](ilsvrc2012/ILSVRC2012_img_val). 
Run script [predict_local.py](scripts/predict/predict_local.py) to obtain predicted WNIDs. 
The predictions will go under [ilsvrc2012/processed/predictions](ilsvrc2012/processed/predictions).
Next run script [evaluate_predictions_local.py](scripts/postprocess/evaluate_predictions_local.py) 
with default parameters. This will add one new column per model to 
[ILSVRC2012_val.parquet](ilsvrc2012/ILSVRC2012_val.parquet) 
that shows whether the predicted WNID for each sample is correct or not.
The average of these columns give accuracy on ILSVRC validation set.

## Compare accuracy on LAIONet and ILSVRC2012
Follow notebook 
[evaluate_predictions_on_subset_substring_matched.ipynb](notebooks/evaluate_predictions_on_subset_substring_matched.ipynb)
to obtain equally weighted and LAION weighted accuracy and compare them on LAIONet and ILSVRC.
This will reproduce results of Section 3.1 in the paper.

## Calculate CLIP zero-shot accuracy
Use scripts [calc_and_store_clip_image_to_all_queries_similarities.py](scripts/calcsimilarity/calc_and_store_clip_image_to_all_queries_similarities.py)
and [calc_and_store_clip_image_to_all_queries_similarities_local.py](scripts/calcsimilarity/calc_and_store_clip_image_to_all_queries_similarities_local.py)
to calculate CLIP image to all synsets similarities for LAIONet and ILSVRC validation set. 
Then follow notebook [calc_clip_zero_shot_accuracy.ipynb](notebooks/calc_clip_zero_shot_accuracy.ipynb).


# Find and compare intra-class similarity
## Find intra-class similarity of LAIONet
Run the following command to find the intra-class similarity of LAIONet images:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities.py \
--labels_file_name "wnid2laionindices(substring_matched).pkl" \
--method substring_matched_filtered \
--do_sample --n_sample 100 \
--n_process_download 16 \
--gpu_id 0
```
This will calculate one pairwise similarity matrix per each WNID in LAIONet 
and save it in [laion400m/processed/clip_image_similarities](laion400m/processed/clip_image_similarities).
Since some classes have many images, we recommend using `--do_sample` and keep only a couple of images per each class.
We recommend using multiple processes for downloading LAIONet images on the fly by setting `--n_process_download`.
We also recommend using a single GPU for this command.

## Find intra-class similarity of ILSVRC2012
Make sure you have downloaded ILSVRC validation images and 
placed them under [ilsvrc2012/ILSVRC2012_img_val](ilsvrc2012/ILSVRC2012_img_val).
Run the following command to find the intra-class similarity of ILSVRC:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities_local.py \
--images_path "ilsvrc2012/ILSVRC2012_img_val" \
--dataframe_path "ilsvrc2012/ILSVRC2012_val.parquet" \
--index2wnid_path "ilsvrc2012/processed/labels/imagename2wnid.pkl" \
--gpu_id 0
```
The results will be in the same format as LAIONet intra-class similarities and go to the same place 
[laion400m/processed/clip_image_similarities](laion400m/processed/clip_image_similarities).

## Find intra-class similarity of ImageNet-V2
First, make sure you have downloaded and preprocessed ImageNet-V2 as explained [here](imagenetv2/readme.md).
Then run the below commands three times
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities_local.py \
--images_path "imagenetv2/imagenetv2-[version]" \
--dataframe_path "imagenetv2/imagenetv2-[version].parquet" \
--index2wnid_path "imagenetv2/processed/labels/imagenetv2-[version]-imagename2wnid.pkl" \
--gpu_id 0
```
where `version` takes three values `matched-frequency`, `threshold0.7`, and `top-images`.
The results will be in the same format as LAIONet intra-class similarities and go to the same place 
[laion400m/processed/clip_image_similarities](laion400m/processed/clip_image_similarities).

## Compare intra-class similarity
Follow notebook [evaluate_distribution_of_similarities.ipynb](notebooks/evaluate_distribution_of_similarities.ipynb).
This will reproduce results of Sections 3.2 and 4.1 in the paper.


# Evaluate text-synset similarity of ImageNet-Captions
In Section 4.2 of the paper, we evaluated text to synset similarity of ImageNet-Captions and showed 
text alone cannot explain why an image is selected into ImageNet. We reproduce the results here.

First, make sure you have downloaded and preprocessed ImageNet-Captions following 
[this](imagenet-captions/readme.md) guideline.

Next, run the following command to calculate text-synset similarity:
```shell
python scripts/calcsimilarity/calc_and_store_clip_text_to_query_similarities_local.py \
--dataframe_path "imagenet-captions/imagenet_captions.parquet" \
--index2wnid_path "imagenet-captions/processed/labels/icimagename2wnid.pkl" \
--query_type name_def \
--gpu_id 0
```
This will add a column `text_to_name_def_wnid_similarity` to `imagenet-captions.parquet`. 

Finally, follow notebook [evaluate_distribution_of_similarities.ipynb](notebooks/evaluate_distribution_of_similarities.ipynb)
to reproduce results of Sections 4.2.


# Recreate ImageNet by searching LAION for most similar captions
In Section 4.3 of the paper, we recreated ImageNet from LAION images which have most similar texts to ImageNet captions.
We showed this new dataset does not resemble the original ImageNet. We reproduce the results here.

First, make sure you have downloaded and preprocessed ImageNet-Captions following 
[these](imagenet-captions/readme.md) steps.

To search LAION, we first need to create an index for its texts. 
You can find a 16 GB FAISS index processed by us [here](https://drive.google.com/drive/folders/1otQEdXWPaPbayJZoYWhizAcMvYNFQGD_?usp=sharing) 
or follow the next steps to obtain your index with desired properties:
1. Run [calc_and_store_clip_text_embeddings.py](scripts/searchtext/calc_and_store_clip_text_embeddings.py)
to obtain an initial set of LAION text embeddings.
2. Run [build_initial_text_index.py](scripts/searchtext/build_initial_text_index.py) to initiate a FAISS index.
3. Run [calc_and_store_clip_text_embeddings_in_faiss_index.py](scripts/searchtext/calc_and_store_clip_text_embeddings_in_faiss_index.py)
repetitively on all or some of the LAION-400M 32 parts to update the index.
Whether you downloaded our processed index or created your own index, you can use
notebook [demo_query_faiss_index.ipynb](notebooks/demo_query_faiss_index.ipynb) to test the index.

Given a text index, run the following command to search the index for most similar texts to ImageNet captions:
```shell
python scripts/searchtext/find_most_similar_texts_to_texts.py \
--dataframe_path "imagenet-captions/imagenet_captions.parquet" \
--gpu_id 0
```
If you created your own index, change the default values of `--indices_path` and `--faiss_index_path` accordingly.
We recommend using a single GPU for this step. 
The results will go under [laion400m/processed/ilsvrc_labels](laion400m/processed/ilsvrc_labels).

To create a dataset from search results, run the following two commands:
```shell
python scripts/preprocess/find_wnid_to_laion_indices_local.py

python scripts/createdataset/subset_laion.py \
--labels_filter "wnid2laionindices(subset_ic_most_sim_txttxt).pkl" \
--method imagenet_captions_most_similar_text_to_texts \
--self_destruct
```
The `--method` parameter in the second command is solely for the consistent naming purpose.
Setting `--self_destruct` ensures the original LAION files will be removed from the disk after process completion.

Now that we have created the new dataset, run the following command to obtain its intra-class similarity:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities.py \
--labels_file_name "wnid2laionindices(subset_ic_most_sim_txttxt).pkl" \
--method imagenet_captions_most_similar_text_to_texts \
--do_sample --n_sample 100 \
--n_process_download 16 \
--gpu_id 0
```

Finally, follow notebook [evaluate_distribution_of_similarities.ipynb](notebooks/evaluate_distribution_of_similarities.ipynb)
to reproduce results of Sections 4.3.
