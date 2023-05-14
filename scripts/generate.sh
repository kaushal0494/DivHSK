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
