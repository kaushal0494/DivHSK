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
