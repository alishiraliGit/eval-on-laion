This is the accompanying code of "What Makes ImageNet Look Unlike LAION."
(<a href="https://openreview.net/pdf?id=IrBYuh9W3T" target="_blank">link to the paper</a>)

This page explains how to 
- [Download and use LAIONet](#download-and-use-laionet).
- [Create LAIONet from scratch](#create-laionet), possibly with your choice of parameters (Section 2 of the paper).
- [Find and compare the accuracy](#find-and-compare-accuracy) of various models on LAIONet and ImageNet ILSVRC2012 (Section 3.1). 
- [Find and compare intra-class similarity](#find-and-compare-intra-class-similarity) of LAIONet and ImageNet ILSVRC2012 (Sections 3.2). 
- Reproduce the results of experiments in Sections 4 of the paper.

# Download and use LAIONet
LAIONet is a research artifact intended to highlight the differences between LAION and ImageNet. 
Our goal was not to provide a new benchmark or a new training set. LAIONet includes confidently sampled images, but the labels are not reviewed by humans and may contain limited mistakes.
Therefore, please use LAIONet cautiously. You may also want to follow next sections to create your own version of LAIONet
with possibly different thresholds.

There are two major decisions in creating _a LAIONet_:
(a) What text encoder to use for calculating textual similarity of the image captions and synset texts,
(b) How to sample based on the calculated similarities. 
Our main results in the paper are based on the text encoder of `clip-vit-base-patch32` and by conservatively including samples with textual similarity above 0.82.
However, post December 2024, we have also included versions with `all-mpnet-base-v2` as the text encoder (Appendix A of the paper)
and an imagenet-like sampling where for each class we sample top 50 images (Appendix B of the paper). 

The above versions of the LAIONet can readily be found and downloaded in the repository:
<table>
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2"></th>
    <th colspan="2">Sampling</th>
  </tr>
  <tr>
    <th>Thresholding</th>
    <th>Top 50</th>
  </tr>
  <tr>
    <th rowspan="2">Text encoder</th>
    <td><code>clip-vit-base-patch32</code></td>
    <td><a href="https://drive.google.com/file/d/1Vlo3qN0TH0q3NNmAF9NxBAMaCleP1HxV/view?usp=drive_link">LAIONet(clip_0.82)</a> [for paper's main results]
+ <a href="laion400m/processed/ilsvrc_labels/wnid2laionindices(subset_sm_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)).pkl">labels dict</a>
</td>
    <td><a href="https://drive.google.com/file/d/1t1tUvJOF3EV0uhCDVqEeYjc72ZcBBaN_/view?usp=drive_link">LAIONet(clip_top50)</a> [Appendix B]
+ <a href="laion400m/processed/ilsvrc_labels/wnid2laionindices(subset_sm_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)_skimmed).pkl">labels dict</a>
</td>
  </tr>
  <tr>
    <td><code>all-mpnet-base-v2</code></td>
    <td><a href="https://drive.google.com/file/d/15nt4_Mj98I-ILii1bD358zqQskt_dGMX/view?usp=drive_link">LAIONet(mpnet_0.58)</a> [Appendix A]
+ <a href="laion400m/processed/ilsvrc_labels/wnid2laionindices(subset_sm_filt(text_to_name_def_wnid_similarity_all-mpnet-base-v2)).pkl">labels dict</a>
</td>
    <td><a href="https://drive.google.com/file/d/1PUkgr7mC5Qs9pwiFtEYY0OlLcAbwXlCD/view?usp=drive_link">LAIONet(mpnet_top50)</a> [Appendix B]
+ <a href="laion400m/processed/ilsvrc_labels/wnid2laionindices(subset_sm_filt(text_to_name_def_wnid_similarity_all-mpnet-base-v2)_skimmed).pkl">labels dict</a>
</td>
  </tr>
</table>

You may want to use `pd.read_parquet` command to read the the above files. 
The images are under their copyright and should be downloaded or processed on the fly (as the codes do in this repository).
Each LAIONet dataset above contains the following columns:
- Main columns:
  - `URL`: The URL of the image.
  - `TEXT`: The caption of the image.
  - `name_def_wnid`: The textual representation of the instance created by concatenating the name
and definition of its associated synset. This is called synset text in the paper and _query_ in the code.
  - `wnid_is_in_recognized_text`: Whether the class name is found in the image recognized text. 
We drop the rows where this column is `True` in our evaluations.
- Other columns:
  - `SAMPLE_ID`: The original ID assigned to each instance in LAION-400M. We will not use these IDs.
  - `HEIGHT`, `WIDTH`, and `LICENSE` of the images.
  - `NSFW`: Whether the images are safe for work. Copied from LAION-400M. Post December 2024, we have only retained `NSFW=UNLIKELY` in the dataset.
  - `similarity`: The CLIP similarity of the image and caption as reported by LAION-400M developers. 
  We use a different version of CLIP which is not consistent with theirs, 
  which we obtain by running [calc_and_store_clip_image_to_text_similarities.py](scripts/calcsimilarity/calc_and_store_clip_image_to_text_similarities.py).
  - `text_to_name_def_wnid_similarity_[text_encoder_ver]`: Textual similarity of the caption and synset text. In other words, the similarity of `TEXT` and
`name_def_wnid` columns. In the paper, this is referred to as text-to-[name: def] similarity. 
As we discuss in the paper and [later](#2-calculate-laion-text-to-synset-text-similarity) in this readme,
there are multiple choices for `text_encoder_ver`. We use `clip-vit-base-patch32` for our main results
but also report the results for `all-mpnet-base-v2` as well.
Note that this similarity is how we filter out lowe quality matches to create LAIONet.
  - `recognized_text`: The output of running a text detector and a text recognition tool on the image.
  - `top_[k]_is_correct_[predictor]`: Whether a predictor's top k predictions include the correct label.
These columns are not part of LAIONet but we have kept them to simplify reproducibility of the results.
We only explore top 1 and top 5. For a list of predictors that we will use for evaluation
see [ilsvrc_predictors.py](core/ilsvrc_predictors.py).

Note that in order to reproduce the results of the paper using the above datasets,
you need to download them and place them under [laion400m](laion400m) folder.

The labels of LAIONet(s) in the above table are pickled, so you may want to use `pickle.load` to load the labels. 
This will load a Python dictionary with a WordNet ID as key
and a list of indices as the value. These indices are referring to the index of LAIONet dataset. 
WordNet IDs are unique IDs assigned to each synset in WordNet.
Look at [this](ilsvrc2012/ILSVRC2012_synsets.txt) file to see what lemmas are associated with each WordNet ID.

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
Creating LAIONet involves four steps: 
1. Substring match LAION texts with the lemmas of ILSVRC 1000 synsets (every synset can have multiple lemmas). 
2. Using CLIP text encoder, or your favorite text encoder, 
calculate the similarity of LAION texts to the textual representations of their matched synsets.
3. Select instances with high textual similarity to create LAIONet.
4. Take a few additional measures to ensure the quality of labeling.

In the following, we explain the commands and execution details for each step.

## 1. Substring match LAION texts with lemmas
Run script [label_laion.py](scripts/createdataset/label_laion.py) to substring match LAION texts with lemmas in 1000 
ILSVRC synsets. The script requires a dictionary, mapping lemmas to their WNIDs. 
We recommend using the [default option](ilsvrc2012/processed/lemma2wnid(unique_in_ilsvrc_ignored_empty_wnids).pkl) 
which only includes lemmas that are unique across synsets 
(this default option itself is generated by running [this](scripts/preprocess/preprocess_ilsvrc_classes.py)). 
The script should be run for each of the 32 parts of LAION-400M.
The parameter `--laion_part` determines the part, taking a value from 0 to 31. 
It also allows parallel processing by setting `--n_process`. 
Make sure to set `--self_destruct` if you want to remove the downloaded LAION part after the process is done. 
Use script [label_laion_parallel.py](scripts/createdataset/label_laion_parallel.py) 
to run on multiple LAION parts in parallel. 
Look at the scripts for further arguments and help.

Using default parameters, the previous scripts save two dictionaries per each part 
in folder [laion400m/processed/ilsvrc_labels](laion400m/processed/ilsvrc_labels). 
These dictionaries are `wnid2laionindices(substring_matched_part[x]).pkl` and 
`lemma2laionindices(substring_matched_part[x]).pkl`, which map each WNID or the lemmas
of a WNID to LAION indices for part `x`.  
Use the following command to extract a dataset from all LAION instances that are matched to at least one lemma:
```shell
python scripts/createdataset/subset_laion.py \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--save_prefix subset_sm \
--self_destruct
```
The `--save_prefix` argument is only for consistent naming throughout the process and `--self_destruct` ensures 
LAION original files will be removed from the disk by process completion.
This command will create a large dataset (`subset_sm_part-00000-to-part00031 ... .parquet`) under folder [laion400m](laion400m) 
requiring ~4 GB of the disk. 

## 2. Calculate LAION text to synset text similarity
The first step to calculate textual similarity of a LAION instance and a synset is to create a textual representation
for the synset. Generally, we call such a representation a query.
Following the paper, we recommend concatenating the name of the synset and its short definition. 
This corresponds to `QueryType.NAME_DEF`. Look at [queries.py](core/queries.py) for a list of possible queries.

After defining a textual representation for the synset, we need to define a similarity measure.
We encode the texts and measure the cosine similarity of the encoded texts.
There are multiple text encoders available at this step. 
By default, we use the text embedding of a base CLIP model (`clip-vit-base-patch32`). 
But any version of Bert listed on [Hugging Face](https://huggingface.co/bert-base-uncased) or 
any OpenAI CLIP implementation on Hugging Face or 
a [SentenceTransformer](https://huggingface.co/sentence-transformers) can be used by specifying `--text_encoder_ver` in the following.
For example, we have used an MPNet sentence encoder (`all-mpnet-base-v2`) to generate a new version of LAIONet in Appendix A of the paper.

Run the following command to calculate LAION text to synset text (query) similarity 
for the dataset we created in the previous step:
```shell
python scripts/calcsimilarity/calc_and_store_text_to_query_similarities.py \
--prefix subset_sm \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--query_type name_def \
--query_key wnid \
--text_encoder_ver "clip-vit-base-patch32" \
--gpu_id 0
```
The above command allows specifying `--gpu_id` or `no_gpu`. Look at the script for further arguments and help.
We recommend using a GPU for this task to accelerate the calculation. 
Upon completion, the above command adds two new columns to the dataset: 
The first column `[query_type]_[query_key]` stores the query and 
the second column `text_to_[query_type]_[query_key]_similarity_[text_encoder_ver]` stores similarity values. 
We drop LAION instances which have lemmas from multiple synsets at this step.

## 3. Select LAION instances with high similarity to their synsets
Run the following command to filter out LAION instances with low textual similarity to their matched synsets:
```shell
python scripts/createdataset/sample_laion_from_most_similars.py \
--load_prefix subset_sm \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--similarity_col "text_to_name_def_wnid_similarity_clip-vit-base-patch32" \
--similarity_th 0.82 \
--remove_nsfw
```
We followed the paper and set the minimum required similarity to 0.82. 
Follow notebook [choose_text_query_similarity_threshold.ipynb](notebooks/choose_text_query_similarity_threshold.ipynb)
to see this is a conservative choice and try out other values.
Setting `--remove_nsfw` only keeps instances labeled to be safe for work. 
Upon completion, the script saves a new dataset that starts with `subset_sm_filt([similarity_col])` under [laion400m](laion400m). 
This command also saves a new dictionary `wnid2laionindices(subset_sm_filt([similarity_col])).pkl` in [laion400m/processed/ilsvrc_labels](laion400m/processed/ilsvrc_labels) 
which maps each WNID to the indices of the new dataset. 

Prior to December 2024, we used to call the dataset generated above using `clip-vit-base-patch32` as the text encoder, LAIONet.
However, in December 2024, we added a column to the dataset to mark the images that have their class name written and excluded these from evaluations, which we discuss below.


## 4. Additional measures 
We first run [download_and_recognize_text.py](scripts/recognizetext/download_and_recognize_text.py), 
which uses a text detector and a text recognition tool to read the texts in the images. This will add a column `recognized_text` to the dataset.
We then run [evaluate_recognize_texts.py](scripts/postprocess/evaluate_recognized_texts.py) to mark the rows where 
the class name is found in the image. We record this in a new column `wnid_is_in_recognized_text`. 
We will drop the rows where `wnid_is_in_recognized_text = True` in our evaluations.

In Appendix B of the paper, we also investigated a more conservative version of LAIONet
where for each class we only keep the top 50 images of the original LAIONet. 
To get this version, run [skim_subset_laion.py](scripts/createdataset/skim_subset_laion.py). 
This will save a new dataset with `_skimmed` appended to the prefix.


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
--prefix "subset_sm_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)" \
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
In this case, we have utilized 16 processes for image downloading and 1 process for predicting their labels.
It is worth noting that parallelization greatly enhances the download process. 
Setting number of processes beyond the number of your processor cores is not recommended.
Additionally, we highly recommend using a single GPU to expedite model evaluation. 
Sometimes downloading is faster than prediction.
The parameter `--pred_max_todo` determines the maximum number of batches downloaded but not predicted. 
After this threshold, the program pauses the download and wait for the prediction to catch up. 
To change the batch size look at `ILSVRCPredictorsConfig` in [configs.py](configs.py).
We also recommend using teh wrapper [download_and_predict_wrapper.py](scripts/predict/download_and_predict_wrapper.py)
for better handling of the exceptions if you're working with default parameters.

The above command saves one `.csv` file for each model in 
[laion400m/processed/ilsvrc_predictions](laion400m/processed/ilsvrc_predictions). 
The file comes with the name of the model and date and contains predicted WNIDs.
To evaluate these predictions, use 
```shell
python scripts/postprocess/evaluate_predictions.py \
--prefix "subset_sm_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)" \
--predictors selected 
```
This command will add two binary columns for each predictor to LAIONet that show
whether top 1 and top 5 predictions of that predictor were correct. 
It also allows selecting specific version of predictions by setting `--predictions_ver`. 
Look at the script for further options and guidance.

## Find accuracy on ILSVRC2012 validation set
Download the images of ILSVRC2012 validation set from https://www.image-net.org/download.php and put them 
under folder [ilsvrc2012/ILSVRC2012_img_val](ilsvrc2012/ILSVRC2012_img_val). 
Run script [predict_local.py](scripts/predict/predict_local.py) to obtain predicted WNIDs. 
The predictions will go under [ilsvrc2012/processed/predictions](ilsvrc2012/processed/predictions).
Next run script [evaluate_predictions_local.py](scripts/postprocess/evaluate_predictions_local.py) 
with default parameters. This will add two new columns per model to 
[ILSVRC2012_val.parquet](ilsvrc2012/ILSVRC2012_val.parquet) 
that show whether the top 1 and top 5 predicted WNIDs for each sample are correct or not.

## Compare accuracy on LAIONet and ILSVRC2012
Follow notebook 
[plot_accuracy_of_laionet_vs_imagenet.ipynb](notebooks/plot_accuracy_of_laionet_vs_imagenet.ipynb)
to obtain equally weighted and LAION weighted accuracy and compare them on LAIONet and ILSVRC.
This will reproduce results of Section 3.1 in the paper.

## Calculate CLIP zero-shot accuracy
Use scripts [calc_and_store_clip_image_to_all_queries_similarities.py](scripts/calcsimilarity/calc_and_store_clip_image_to_all_queries_similarities.py)
and [calc_and_store_clip_image_to_all_queries_similarities_local.py](scripts/calcsimilarity/calc_and_store_clip_image_to_all_queries_similarities_local.py)
to calculate CLIP image to all synsets similarities for LAIONet and ILSVRC validation set. 
Then use [evaluate_clip_zero_shot_predictions_on_laion.py](scripts/postprocess/evaluate_clip_zero_shot_predictions_on_laion.py)
and [evaluate_clip_zero_shot_predictions_local.py](scripts/postprocess/evaluate_clip_zero_shot_predictions_local.py)
to add CLIP top 1 and top 5 accuracy columns to LAIONet.
We have used these numbers in Section 2 of the paper.


# Find and compare intra-class similarity
## Find intra-class similarity of LAIONet
Run the following command to find the intra-class similarity of LAIONet images:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities.py \
--prefix "subset_sm_large_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)" \
--do_sample --n_sample 100 \
--n_process_download 16 \
--gpu_id 0
```
This will calculate one pairwise similarity matrix per each WNID in the dataset 
and save it in [laion400m/processed/clip_image_similarities](laion400m/processed/clip_image_similarities).
Since some classes have many images, we recommend using `--do_sample` and keep only a couple of images per each class.
We recommend using multiple processes for downloading LAIONet images on the fly by setting `--n_process_download`.
We also recommend using GPU for this command.
The above command by default uses `clip-vit-base-patch32` to get image embeddings.
However, you can use other [image encoders](core/image_encoders.py) through `--image_encoder_ver [encoder_ver]`.
In this case, we recommend using `--use_encoder_ver_in_file_name` to avoid overwriting prior results. 
Appendix H of the paper shows consistent results regardless of the choice of image encoder. 

## Find intra-class similarity of ILSVRC2012
Make sure to download ILSVRC validation images and 
place them under [ilsvrc2012/ILSVRC2012_img_val](ilsvrc2012/ILSVRC2012_img_val).
Run the following command to find the intra-class similarity of ILSVRC:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities_local.py \
--images_path "ilsvrc2012/ILSVRC2012_img_val" \
--dataframe_path "ilsvrc2012/ILSVRC2012_val.parquet" \
--index2wnid_path "ilsvrc2012/processed/labels/imagename2wnid.pkl" \
--gpu_id 0
```
The results go to the same place 
[laion400m/processed/clip_image_similarities](laion400m/processed/clip_image_similarities).
As before, you can use `--image_encoder_ver [encoder_ver]` to use other image encoders here.

## Compare intra-class similarity
Follow notebook [compare_intra_class_similarites_of_laionet_vs_others.ipynb](notebooks/compare_intra_class_similarites_of_laionet_vs_others.ipynb).
to reproduce results of Sections 3.2.


# Reproducing Section 4.1: _A Weaker Image-To-Selection Link Makes ImageNet More Like LAIONet_

In Section 4.1 of the paper, we looked into three versions of ImageNet-V2 and showed
that the extra intra-class diversity of LAIONet is achievable from less stringent human annotation. 

Make sure to download and preprocess ImageNet-V2 as explained [here](imagenetv2/readme.md) before proceeding.

## Find and compare intra-class similarity
We have already calculated intra-class similarity of LAIONet and ILSVRC. 
To find intra-class similarities of ImageNet-V2 versions run the following command three times:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities_local.py \
--images_path "imagenetv2/imagenetv2-[version]" \
--dataframe_path "imagenetv2/imagenetv2-[version].parquet" \
--index2wnid_path "imagenetv2/processed/labels/imagenetv2-[version]-imagename2wnid.pkl" \
--gpu_id 0
```
Here, `version` takes three values `matched-frequency`, `threshold0.7`, and `top-images`.
The results will go under
[laion400m/processed/clip_image_similarities](laion400m/processed/clip_image_similarities).

Follow notebook [evaluate_distribution_of_similarities.ipynb](notebooks/evaluate_distribution_of_similarities.ipynb) 
to regenerate the plots of Section 4.1.

## Find and compare accuracy
We have also already calculated accuracy of various models on LAIONet and ILSVRC.
To find the accuracy on ImageNet-V2, run
```shell
python scripts/predict/predict_local.py \
--images_path "imagenetv2/imagenetv2-[ver]" \
--dataframe_path "imagenetv2/imagenetv2-[v2].parquet" \
--save_path "imagenetv2/processed/predictions" \
--predictors selected \
--gpu_id 0
```
where `ver` can take three values of `matched-frequency`, `threshold0.7`, and `top-images`.
The use notebook [plot_accuracy_of_laionet_vs_imagenetv2.ipynb](notebooks/plot_accuracy_of_laionet_vs_imagenetv2.ipynb)
to generate the plots.

# Reproducing Section 4.2: _Introducing an Image-To-Selection Link Makes LAIONet More Like ImageNet_
In Section 4.2, we constructed new datasets out of LAION where we required an included sample to have image-to-[name: def] similarity greater than a threshold and text-to-[name: def] similarity also greater than another
threshold, while controlling the total number of selected samples to be similar to
our original version of LAIONet. We reproduce the results of this section here which requires multiple steps.

## Filter based on textual similarity
We use a less stringent textual similarity threshold in creating the datasets of Section 4.2.
We follow very similar steps as our [steps](#create-laionet) to create LAIONet itself: 
First, if you haven't already, follow [this](#1-substring-match-laion-texts-with-lemmas)
to substring match LAION texts with lemmas which creates a large dataset. 
Then create a copy of this dataset as follows
```shell
cp laion400m/subset_sm_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet \
laion400m/subset_sm_large_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet
```
Since the smallest textual similarity threshold in this section is 0.72, 
run the following command to filter out samples with lower similarity:
```shell
python scripts/createdataset/sample_laion_from_most_similars.py \
--load_prefix subset_sm_large \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--similarity_col "text_to_name_def_wnid_similarity_clip-vit-base-patch32" \
--similarity_th 0.72 \
--remove_nsfw
```

## Find multimodal similarity
To calculate image-to-[name: def] similarity, run
```shell
python scripts/calcsimilarity/calc_and_store_clip_image_to_query_similarities.py \
--prefix "subset_sm_large_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)" \
--labels_filter "wnid2laionindices(subset_sm_large_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)).pkl" \
--query_type name_def \
--query_key wnid \
--n_process_download 16 \
--save_freq 1000
```
This will add a new column `image_to_name_def_wnid_similarity_clip-vit-base-patch32` to the dataset.
Since this process is time-consuming, you can use `--from_iloc` and `--to_iloc` arguments 
and later use [this](scripts/postprocess/copy_col_from_src_to_target.py) script to aggregate the results.

## Filter based on multimodal similarity
The four datasets (A, B, C, D) in Section 4.3 use different textual and image-to-text similarity thresholds.
Let t be the textual threshold for text-to-[name: def] similarity 
and i be the threshold for image-to-[name: def] similarity. 
To create a dataset with (t, i) thresholds, we first copy our latest dataset:
```shell
cp "laion400m/subset_sm_large_filt(text_to_name_def_wnid_similarity_clip-vit-base-patch32)_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet" \
"laion400m/subset_sm_filt(mm_[t]_[i])_part-00000-to-part00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
```
Then we apply textual and multimodal filterings:
```shell
python scripts/createdataset/sample_laion_from_most_similars.py \
--load_prefix "subset_sm_filt(mm_[t]_[i])" \
--save_prefix "subset_sm_filt(mm_[t]_[i])" \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--similarity_col "text_to_name_def_wnid_similarity_clip-vit-base-patch32" \
--similarity_th [t] \
--remove_nsfw \
--no_safe

python scripts/createdataset/sample_laion_from_most_similars.py \
--load_prefix "subset_sm_filt(mm_[t]_[i])" \
--save_prefix "subset_sm_filt(mm_[t]_[i])" \
--labels_filter "wnid2laionindices(substring_matched_part*).pkl" \
--similarity_col "image_to_name_def_wnid_similarity_clip-vit-base-patch32" \
--similarity_th [i] \
--remove_nsfw \
--no_safe
```
We repeat the above steps for every combination of (t, i) that's discussed in the paper.
To see how we have chosen these thresholds, see notebook 
[choose_multimodal_filtering_similarity_thresholds.ipynb](notebooks/choose_multimodal_filtering_similarity_thresholds.ipynb).

## Find and compare intra-class similarity across datasets
The last step to reproduce the results of Section 4.2 is to find intra-class similarity of the four created datasets, by running
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities.py \
--prefix "subset_sm_filt(mm_[t]_[i])" \
--do_sample --n_sample 100 \
--n_process_download 16 \
--gpu_id 0
```
Note that we repeat teh above four times for all the combinations of (t, i).
We have also reported consistent results using other image encoders in Appendix H.
To reproduce those results, use `--image_encoder_ver [encoder_ver]` along with `--use_encoder_ver_in_file_name` arguments.
Finally, use notebook [compare_intra_class_similarites_of_laionets.ipynb](notebooks/compare_intra_class_similarites_of_laionets.ipynb) to regenarate the same plots.

# Reproducing Section 4.3: _Text Alone Cannot Explain Why an Image Is Selected Into ImageNet_
In Section 4.3 of the paper, we evaluated text to synset similarity of ImageNet-Captions and showed 
text alone cannot explain why an image is selected into ImageNet. We reproduce the results here.

First, make sure you have downloaded and preprocessed ImageNet-Captions following 
[this](imagenet-captions/readme.md) guideline.

Second, run the following command to calculate text-synset similarity:
```shell
python scripts/calcsimilarity/calc_and_store_clip_text_to_query_similarities_local.py \
--dataframe_path "imagenet-captions/imagenet_captions.parquet" \
--index2wnid_path "imagenet-captions/processed/labels/icimagename2wnid.pkl" \
--query_type name_def \
--gpu_id 0
```
This will add a column `text_to_[query_type]_wnid_similarity` to `imagenet-captions.parquet`. 

Third, in addition to the similarity of the text to its matched synset, we also need the similarity of the text to other synsets.
Copy `imagenet-captions.parquet` to a new file `imagenet_captions_with_sims_to_all_queries.parquet`. Then run
```shell
python scripts/calcsimilarity/calc_and_store_clip_text_to_all_query_similarities_local.py \
--dataframe_path "imagenet-captions/imagenet_captions_with_sims_to_all_queries.parquet" \
--query_type name_def \
--gpu_id 0
```
This will add one column `text_to_[query_type]_[WNID]_similarity` per each WNID of ILSVRC2012
to `imagenet_captions_with_sims_to_all_queries.parquet`. Note that the resulting file will get ~2.8 GB of the disk.

Finally, follow notebook [evaluate_distribution_of_similarities.ipynb](notebooks/evaluate_distribution_of_similarities.ipynb)
to regenerate plots of Sections 4.3.


# Reproducing Section 4.4: _ImageNet, Had It Been Created Solely Searching Texts, Does Not Resemble Current ImageNet_
In Section 4.4 of the paper, we recreated ImageNet from LAION images which have most similar texts to ImageNet captions.
We showed this new dataset does not resemble the original ImageNet. We reproduce the results here.

Before getting started, make sure you have downloaded and preprocessed ImageNet-Captions following 
[these](imagenet-captions/readme.md) steps.

## Create an index for LAION texts
To search LAION, we first need to create an index for its texts. 
You can find a 16 GB FAISS index processed by us [here](https://drive.google.com/drive/folders/1qtfkQPYwUo9k2wTzuBbeWY6fE77l71oH?usp=sharing) 
or follow the next steps to obtain your index with desired properties:
1. Run [calc_and_store_clip_text_embeddings.py](scripts/searchtext/calc_and_store_clip_text_embeddings.py)
to obtain an initial set of LAION text embeddings.
2. Run [build_initial_text_index.py](scripts/searchtext/build_initial_text_index.py) to initiate a FAISS index.
3. Run [calc_and_store_clip_text_embeddings_in_faiss_index.py](scripts/searchtext/calc_and_store_clip_text_embeddings_in_faiss_index.py)
repetitively on all or some of the LAION-400M 32 parts to update the index.
Whether you downloaded our processed index or created your own index, you can use
notebook [demo_query_faiss_index.ipynb](notebooks/demo_query_faiss_index.ipynb) to test the index.

## Create a new dataset of LAION instances with the most similar texts to the ImageNet captions
Given a text index, run the following command to search the index for most similar texts to ImageNet captions:
```shell
python scripts/searchtext/find_most_similar_texts_to_texts.py \
--dataframe_path "imagenet-captions/imagenet_captions.parquet" \
--gpu_id 0
```
If you created your own index, change the default values of `--indices_path` and `--faiss_index_path` accordingly.
We recommend using a GPU for this step. 
This commands result will save `icimagename2laionindices.pkl` and `icimagename2sims.pkl`
under [laion400m/processed/ilsvrc_labels](laion400m/processed/ilsvrc_labels).

To create a dataset from search results, first, run
```shell
python scripts/preprocess/find_wnid_to_laion_indices_local.py \
--index2laionindices_path "laion400m/processed/ilsvrc_labels/icimagename2laionindices.pkl" \
--index2wnid_path "imagenet-captions/processed/labels/icimagename2wnid.pkl" \
--save_file_name "wnid2laionindices(subset_ic_most_sim_txttxt).pkl"
```
This will store a mapping from every WNID to LAION indices of the new dataset. Then run
```shell
python scripts/createdataset/subset_laion.py \
--labels_filter "wnid2laionindices(subset_ic_most_sim_txttxt).pkl" \
--save_prefix subset_ic_most_sim_txttxt \
--self_destruct
```
This will create a new dataset out of LAION with the name starting with `subset_ic_most_sim_txttxt` under [laion400m](laion400m).

## Calc. intra-class similarity
Now that we have created the new dataset, run the following command to obtain its intra-class similarity:
```shell
python scripts/calcsimilarity/calc_and_store_intra_class_image_similarities.py \
--prefix subset_ic_most_sim_txttxt \
--do_sample --n_sample 100 \
--n_process_download 16 \
--gpu_id 0
```

## Generate plots
Finally, follow notebook [evaluate_distribution_of_similarities.ipynb](notebooks/evaluate_distribution_of_similarities.ipynb)
to regenerate plots of Sections 4.4.


# Feedback and report issues
We appreciate your feedback and please feel free to reach out to the authors or report issues here.