# Coleridge Initiative - Show US the data Solution


## Introduction

The aim of this competition is to identify critical datasets used in scientific publications using NLP (Natural Language Processing) for automatic search. Utilizing the full text of scientific publications from numerous research areas gathered from [CHORUS](https://www.chorusaccess.org/) publisher members and other sources, you'll identify data sets that the publications' authors used in their work.

![Alt text](Coleridge_Initiative_Solution.png?raw=true "Coleridge Initiative - Show US the Data Solution")

## Disclaimer

* This is our work, we **DO NOT** represent any organization
* There's no reproducibility guarantee for notebook which uses GPU and TPU
* Dataset and generated dataset falls under Coleridge Initiative Terms and Conditions which can be seen on [Kaggle Datasets](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/data)


## Environment List

> Some of the notebook are run on different environment. We use Kaggle GPU on training the model and submitting the code to get public/private score.

| Environment Name | Description                            |
| ---------------- | -------------------------------------- |
| Kaggle CPU       | 2C/4T CPU, 16GB RAM                    |
| Kaggle GPU       | 2C/4T CPU, 16GB RAM, Nvidia Tesla P100 |
| Kaggle TPU       | 2C/4T CPU, 16GB RAM, TPU v3-8          |


## Notebook Description
### Dataset Scrapper
| Filename           | Link to Kaggle Kernel                                                                      | Environment | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------- |
| 100000-govt-datasets-api-json-to-df.ipynb      | https://www.kaggle.com/mlconsult/100000-govt-datasets-api-json-to-df/         | Kaggle CPU  | The external US Goverment's datasets scrapper by Ken Miller											|
| web-scraping-for-bigger-govt-dataset-list.ipynb      | https://www.kaggle.com/chienhsianghung/coleridge-additional-gov-datasets-22000popular         | Kaggle CPU  | The external US Goverment's datasets scrapper by Chien-Hsiang Hung 		|

### Dataset preprocessing and filter
| Filename           | Link to Kaggle Kernel                                                                      | Environment | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------- |
| ci-create-train-df-with-ext-data.ipynb      | https://www.kaggle.com/mfalfafa/ci-create-train-df-with-ext-data         | Kaggle CPU  | Identify the mention of datasets within scientific publications using literal prediction (filtered scrapped datasets) 		|
| ci-train-ext-dataset-v2.ipynb      | https://www.kaggle.com/mfalfafa/ci-train-ext-dataset-v2/         | Kaggle CPU  | Identify the mention of datasets within scientific publications using literal prediction (filtered scrapped datasets) 		|

### NLP model generator

| Filename           | Link to Kaggle Kernel                                                                      | Environment | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------- |
| coleridge-spacy-model-training.ipynb      | https://www.kaggle.com/mfalfafa/coleridge-spacy-model-training/         | Kaggle TPU  | Create NLP model using Spacy classifier                                                             |
| coleridge-xlm-roberta-base-epoch-1-training.ipynb      | https://www.kaggle.com/mfalfafa/coleridge-xlm-roberta-base-epoch-1-training/         | Kaggle GPU  | Create NLP model for Named Entity Recognition (NER) task using XLM-RoBERTa-base as a pretrained model                                                             |
| sentence-level-analysis-with-transformer-v2.ipynb      | https://www.kaggle.com/mfalfafa/sentence-level-analysis-with-transformer-v2  | Kaggle GPU  | Create NLP model for sentence analysis task using multihead attention-based and MLP model                                                             |

### Final Solution
| Filename           | Link to Kaggle Kernel                                                                      | Environment | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------- |
| coleridge-initiative-final-inference.ipynb      | https://www.kaggle.com/mfalfafa/coleridge-initiative-final-inference         | Kaggle GPU  | Final solution for prediction of mentioned datasets. This notebook used for submission                                                                              |


## Generated Datasets and Models
### Generated Datasets

Datasets are generated using scrapper notebooks. This dataset can be used to make quick submission using final-solution notebook.

| Scrapper/filter file         | Generated dataset                | Link to Kaggle datasets                                                     |
| ------------------ | ----------------------------- | --------------------------------------------------------------------------- |
| 100000-govt-datasets-api-json-to-df.ipynb             | `100000 + govt_datasets_api_json_to_df`      | https://www.kaggle.com/mlconsult/100000-govt-datasets-api-json-to-df/data |
| 100000-govt-datasets-api-json-to-df.ipynb             | `bigger_govt_dataset_list`      | https://www.kaggle.com/mlconsult/bigger-govt-dataset-list |
| web-scraping-for-bigger-govt-dataset-list.ipynb      | `Coleridge additional_gov_datasets_22000popular`          | https://www.kaggle.com/chienhsianghung/coleridge-additional-gov-datasets-22000popular                                  |
| ci-train-ext-dataset-v2.ipynb      | `CI_ext_datasets_found_in_train_v2`          | https://www.kaggle.com/mfalfafa/ci-ext-datasets-found-in-train-v2 |

### Generated Models
| Training file         | Generated model                | Link to Kaggle datasets                                                     |
| ------------------ | ----------------------------- | --------------------------------------------------------------------------- |
| coleridge-xlm-roberta-base-epoch-1-training.ipynb      | `NER model`          | https://www.kaggle.com/mfalfafa/coleridge-xlm-roberta-base-epoch-1-training/data						|
| coleridge-spacy-model-training.ipynb      | `Spacy Classifier model`          | https://www.kaggle.com/mfalfafa/coleridge-spacy-classifier						|
| sentence-level-analysis-with-transformer-v2.ipynb      | `Multihead attention-based model`          | https://www.kaggle.com/mfalfafa/ci-transformers-model-v2		|


## Dependency Datasets

Dependency datasets are used for training the model and submitting the solution. For more details of required version of each python package/library could be seen on **requirements.txt** file.

| Dataset name         | Description		| Link to Kaggle datasets                                                     |
| ------------------ | ------------------ | ----------------------------- |
| Coleridge packages      | Python libraries for datasets, seqeval, tokenizer and transformers | https://www.kaggle.com/tungmphung/coleridge-packages   |
| Kaggle NER utils    | Python file to train BERT model with Named Entity Recognition (NER) task | https://www.kaggle.com/tungmphung/kaggle-ner-utils |


## LB/Leaderboard Score

| Notebook filename | Submission filename    | Public LB   | Private LB  | rank |
| ----------------- | ---------------------- | ----------- | ----------- | -----|
| coleridge-initiative-final-inference.ipynb   | submission.csv         | **0.483** | **0.350** | top 3% |


## Reproducibility Guide

> This guide assume you have necessary files (full dataset provided by Coleridge Initiative and dependency datasets), move it to correct directory path and run it on Kaggle Notebook or another Jupyter Notebook environment. 

1. Run `100000-govt-datasets-api-json-to-df.ipynb` and `web-scraping-for-bigger-govt-dataset-list.ipynb`
2. Run `ci-create-train-df-with-ext-data.ipynb`
3. Run `ci-train-ext-dataset-v2.ipynb`
4. Run `coleridge-spacy-model-training.ipynb`
5. Run `coleridge-xlm-roberta-base-epoch-1-training.ipynb`
6. Run `sentence-level-analysis-with-transformer-v2.ipynb`
7. Run `coleridge-initiative-final-inference.ipynb`
