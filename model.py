import os
import torch
import torch.nn as nn
import sys 

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def copy_layers(src_enc, dest_enc):
    #copy_src_dec_module = nn.ModuleList(src_dec._modules['block'])
    #dest_dec._modules['block'].load_state_dict(copy_src_dec_module.state_dict())
    copy_src_enc_module = nn.ModuleList(src_enc._modules['block'])
    dest_enc._modules['block'].load_state_dict(copy_src_enc_module.state_dict())


def load_model_tokenizer(training_arg):
    tokenizer = AutoTokenizer.from_pretrained(
        training_arg.model_chkpt,
        use_fast=False, 
        cache_dir=training_arg.cache_dir,
    )
    special_tokens = {"eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token, \
        "sep_token": tokenizer.eos_token, "unk_token": tokenizer.unk_token}
    tokenizer.add_special_tokens(special_tokens)

    config = AutoConfig.from_pretrained(
        training_arg.model_chkpt,
        cache_dir=training_arg.cache_dir,
        bos_token_id= tokenizer.bos_token_id,
        eos_token_id= tokenizer.eos_token_id,
        sep_token_id= tokenizer.sep_token_id,
        pad_token_id= tokenizer.pad_token_id,
        unk_token_id= tokenizer.unk_token_id,
        output_hidden_states=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
            training_arg.model_chkpt,
            config=config,
            cache_dir=training_arg.cache_dir,
        )
    mix_model = AutoModelForSeq2SeqLM.from_pretrained(
            training_arg.model_chkpt,
            config=config,
            cache_dir=training_arg.cache_dir,
        )
        
    copy_layers(model.encoder, mix_model.encoder)
    copy_layers(model.encoder, mix_model.keyencoder)

    
    mix_model.save_pretrained("proposed_model")

  
    return mix_model, tokenizer