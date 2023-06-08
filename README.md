# DivHSK


![](https://github.com/kaushal0494/DivHSK/blob/main/divhsk_model.png)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Arko98/Hostility-Detection-in-Hindi-Constraint-2021/blob/main/LICENSE)
[![others](https://img.shields.io/badge/Huggingface-Cuda%2011.1.0-brightgreen)](https://huggingface.co/)
[![others](https://img.shields.io/badge/PyTorch-Stable%20(1.8.1)-orange)](https://pytorch.org/)

## Sample Generation from DivHSK


![](https://github.com/kaushal0494/DivHSK/blob/main/divhsk_gen.png)

## About

This repository contains the source code of the paper titled DIVHSK: Diverse Headline Generation using Self-Attention based Keyword Selection which is accepted in the Findings of the Association of Computational Linguistics (ACL 2023) conference. If you have any questions, please feel free to create a Github issue or reach out to the authors.

## Environment Setup
To set up the environment, use the following conda commands:
```
conda env create --file env.yml
conda activate py38_DivHsk
```
The code was tested with Python=3.8, PyTorch==1.8, and transformers=4.11.

## Training & Generation

- The `MR_HEAD` dataset included in the datasets folder. 
- To get the preprocessed data which is to be used for training and generation, run the jupyter notebook `dataset_Pre_processing.ipynb`
- Run below scripts for model training and headline generation

### Model Training 
```
#!/bin/bash
cd ..

export CUDA_VISIBLE_DEVICES=3
export NCCL_DEBUG=INFO
export NPROC_PER_NODE=1
#export PARENT=`/bin/hostname -s`
export PARENT="base"
export MPORT=1436

#input and output directories
export BASE_DIR='.'
export input_dir="datasets/preprocessed"
export output_dir="outputs/DivHSK"

#model details
export model_type="t5" 
export model_chkpt="t5-base"
export cache_dir='../cache_dir'

python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr="$PARENT" \
    --master_port="$MPORT" "train.py" \
    --train_data ${input_dir}/train_data.tsv \
    --val_data ${input_dir}/val_data.tsv \
    --output_dir ${output_dir} \
    --model_type $model_type \
    --model_chkpt $model_chkpt \
    --max_source_length 128 \
    --max_target_length 32 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --beam_size 5\
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 \
     --num_train_epochs 50\
    --lr_scheduler_type "linear" \
    --logging_steps 30 \
    --save_steps 200 \
    --eval_steps 50 \
    --cache_dir ${cache_dir} \
    --read_n_data_obj -1 \
    --label_smoothing_factor 0.0\
    --do_train \
```
### Generating Headlines

```
#!/bin/bash
cd ..
#setting GPUs to use
export CUDA_VISIBLE_DEVICES=6

# misc. settings
export seed=1453

# model settings
export model_type="t5" 
export model_chkpt="outputs/DivHSK/checkpoint-1450"
#input and output directories
export input_dir="datasets/preprocessed"
export output_dir="outputs/DivHSK"
export gen_file_name="gen.tsv"
export cache_dir='../cache_dir'

python train.py \
    --test_data ${input_dir}/test_data.tsv\
    --output_dir ${output_dir} \
    --model_type ${model_type} \
    --model_chkpt ${model_chkpt} \
    --test_batch_size 32 \
    --max_source_length 128 \
    --max_target_length 32 \
    --length_penalty 0.6 \
    --early_stopping \
    --num_of_return_seq 1 \
    --max_generated_seq_len 100 \
    --min_generated_seq_len 1 \
    --cache_dir ${cache_dir} \
    --top_k 50\
    --top_p 0.95\
    --do_test \
    --read_n_data_obj -1 \
    --gen_file_name ${gen_file_name} \
```

