# DivHSK

<img width="602" alt="DivHSK_Model_Architecture" src="https://github.com/kaushal0494/DivHSK/assets/43181857/82f68c4e-9d3e-439a-a01e-80edc77eac6b">

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Arko98/Hostility-Detection-in-Hindi-Constraint-2021/blob/main/LICENSE)
[![others](https://img.shields.io/badge/Huggingface-Cuda%2011.1.0-brightgreen)](https://huggingface.co/)
[![others](https://img.shields.io/badge/PyTorch-Stable%20(1.8.1)-orange)](https://pytorch.org/)

## Sample Generation from DivHsk

<img width="328" alt="divhsk_results_2" src="https://github.com/kaushal0494/DivHSK/assets/43181857/a2e58f35-26f3-4cc7-aa44-670d5f07c557">

## About

This repository contains the source code of the paper titled DIVHSK: Diverse Headline Generation using Self-Attention based Keyword Selection which is yet to be published in the Findings of the Association of Computational Linguistics (ACL 2023) conference. If you have any questions, please feel free to create a Github issue or reach out to the authors.

## Authors
1) Venkatesh Elangovan (https://github.com/venkateshelangovan) [IIT Hyderabad M.Tech in Artificial Intelligence]
2) Kaushal Kumar Maurya (https://github.com/kaushal0494) [IIT Hyderabad PhD in Computer Science and Engineering]
3) Dr. Maunendra Sankar Desarkar (https://www.iith.ac.in/~maunendra/) [Assistant Professor, Computer Science and Engineering Department, IIT Hyderabad]

## Environment Setup
To set up the environment, use the following conda commands:
```
conda env create --file env.yml
conda activate py38_DivHsk
```
The code was tested with Python=3.8, PyTorch==1.8, and transformers=4.11.

## Training & Generation

- The dataset is organized in the datasets folder. 
- To get the preprocessed data which is to be used for training and generation, run the jupyter notebook 'dataset_Pre_processing.ipynb'

### Step 1: Training 
```
cd scripts
bash train.sh
```
### Step 2: Generate Headlines
Inside scripts directory.
```
bash generate.sh
```


