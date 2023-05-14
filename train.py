import os
from random import shuffle
import sys
import argparse
import json
import numpy as np
from numpy.lib.npyio import save
import copy
from tqdm import tqdm

import opts as opts
import gensim.downloader as api
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
)

from utils import (
    freeze_embeds, 
    freeze_params,
    assert_all_frozen,
)

from model import load_model_tokenizer
from custom_datasets import load_the_dataset
from datasets import load_metric 

#import apex
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

#import socket
#os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
#os.environ['TOKENIZERS_PARALLELISM'] = 'False'

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opts.add_md_help_argument(parser)
    opts.train_opts(parser)

    training_args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    #Create and write config file in output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, 'config.json'), 'w') as f_out:
        json.dump(vars(training_args), f_out)
        f_out.close()
    
    #Set the random seed for deterministic nature
    set_seed(training_args.seed)

    #loading model and tokenizer
    model, tokenizer = load_model_tokenizer(training_args)
    word2vec_model = api.load('word2vec-google-news-300')
    stop_words = stopwords.words('english')

    #Freezing model components 
    logger.info("Total Number of parameters: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if training_args.freeze_embeds:
        freeze_embeds(model)
    #training_args.freeze_keyencoder = True 
    #if training_args.freeze_keyencoder:
    #    print("Fr")
    #    freeze_params(model.get_keyencoder())
    #    assert_all_frozen(model.get_keyencoder())
    if training_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())
    if training_args.freeze_embeds_and_decoder:
        freeze_embeds(model)
        freeze_params(model.get_decoder())
        assert_all_frozen(model.get_decoder())
    logger.info("Total Number of parameters after FREEZE (if any): %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    if training_args.train_data: train_dataset = load_the_dataset(training_args, training_args.train_data, tokenizer, "train")
    if training_args.val_data: val_dataset = load_the_dataset(training_args, training_args.val_data, tokenizer, "val")
    if training_args.test_data: test_dataset = load_the_dataset(training_args, training_args.test_data, tokenizer, "test")
    #print data size:
    if training_args.train_data: logger.info("Training Data Size : %s", len(train_dataset))
    if training_args.val_data: logger.info("Validation Data Size : %s", len(val_dataset))
    if training_args.test_data: logger.info("Test Data Size : %s", len(test_dataset))

    # Save training 
    if training_args.do_train:
        with open(os.path.join(training_args.output_dir, "debug_train_examples.txt"), 'w', encoding='utf8') as debug_fp:
            for example in train_dataset[:50]:
                example_instance = {}
                example_instance['input'] = example['input_ids'].tolist()
                example_instance['attention_mask'] = example['attention_mask'].tolist()
                example_instance['labels'] = example['labels'].tolist()

                example_instance['input_text'] = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
                example_instance['label_text'] = tokenizer.decode(example['labels'], skip_special_tokens=False)

                debug_fp.write(json.dumps(example_instance) + '\n')
            debug_fp.close()
  
        hf_training_args = TrainingArguments(
            output_dir = training_args.output_dir,
            num_train_epochs=training_args.num_train_epochs,
            per_device_train_batch_size=training_args.train_batch_size, 
            per_device_eval_batch_size=training_args.eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=training_args.eval_steps,
            warmup_ratio=training_args.warmup_ratio,
            learning_rate= training_args.learning_rate,
            lr_scheduler_type=training_args.lr_scheduler_type,
            weight_decay=training_args.weight_decay,
            dataloader_num_workers=training_args.num_workers,
            label_smoothing_factor = training_args.label_smoothing_factor,
            fp16=False,
            #logging_dir=os.path.join(training_args.output_dir, "logs"),
            logging_steps=training_args.logging_steps,
            logging_first_step=True,
            save_steps=training_args.save_steps,
            save_total_limit=training_args.save_total_limit,
            load_best_model_at_end=True,
            seed=training_args.seed,
            sharded_ddp=True,
            report_to="tensorboard",
            local_rank=training_args.local_rank,
            overwrite_output_dir="False",
            adafactor=False,
            #resume_from_checkpoint=training_args.resume_from_checkpoint,
        )
        # Transformer library (trainer.py) - Trainer class is instantiated 
        trainer = Trainer(
            model=model, 
            args=hf_training_args,
            train_dataset=train_dataset, 
            eval_dataset=val_dataset, 
            tokenizer=tokenizer,
            word2vecmodel = word2vec_model, 
            stop_words = stop_words,
            loss_multiplier = 0.5,
        )
       # call the method train 
        trainer.train()
        trainer.save_model()
        #torch.save(model.state_dict(), "outputs/myfirst_extd/checkpoint-35000.pt")
    if training_args.do_test:
        with open(os.path.join(training_args.output_dir, 'debug_test_examples.txt'), 'w', encoding='utf8') as f_out:
            for example in test_dataset[:50]:
                example_instance = {}
                example_instance['input'] = example['input_ids'].tolist()
                example_instance['attention_mask'] = example['attention_mask'].tolist()
                example_instance['labels'] = example['labels'].tolist()
                example_instance['labels1'] = example['labels1'].tolist()
                example_instance['labels2'] = example['labels2'].tolist()

                example_instance['input_text'] = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
                example_instance['label_text'] = tokenizer.decode(example['labels'], skip_special_tokens=False)
                example_instance['label_text1'] = tokenizer.decode(example['labels1'], skip_special_tokens=False)
                example_instance['label_text2'] = tokenizer.decode(example['labels2'], skip_special_tokens=False)

                f_out.write(json.dumps(example_instance) + '\n')
            f_out.close()
        
        test_datasets= torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=training_args.test_batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=training_args.num_workers
        )
        """test_output = trainer.predict(
            test_dataset=test_datasets,
            max_length=training_args.max_generated_seq_len,
            num_beams=training_args.beam_size,
            length_penalty=training_args.length_penalty,
            no_repeat_ngram_size=training_args.no_repeat_ngram_size,
        )"""
        
        #tranfering model and tensor to Device
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        with torch.no_grad(), open(os.path.join(training_args.output_dir, training_args.gen_file_name), 'w', encoding='utf8') as f_gen:
            #batch_size = 0 
            #pred_store = {}
            for test_idx, test_instance in tqdm(enumerate(test_datasets), total=len(test_datasets)):
                outputs = model.generate(
                    input_ids=test_instance['input_ids'].to(device),
                    attention_mask=test_instance['attention_mask'].to(device),
                    max_length=training_args.max_generated_seq_len,
                    min_length=training_args.min_generated_seq_len,
                    num_return_sequences=training_args.num_of_return_seq,
                    early_stopping=training_args.early_stopping,
                    pad_token_id=tokenizer.pad_token_id,
                    #num_beams=training_args.beam_size,
                    repetition_penalty=training_args.repetition_penalty,
                    no_repeat_ngram_size=training_args.no_repeat_ngram_size,
                    length_penalty=training_args.length_penalty,
                    # adding some more parameters for top-k and top-p sampling
                    top_k = training_args.top_k, 
                    top_p = training_args.top_p,
                    do_sample = True, 
                ) 
                """
                beam1 = []
                beam2 = []
                beam3 = []
                for i in range(len(outputs)):
                    batch_predictions=tokenizer.batch_decode(outputs[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_references =tokenizer.batch_decode(test_instance['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_predictions = [batch_predictions[i:i + training_args.num_of_return_seq] for i in range(0, len(batch_predictions), training_args.num_of_return_seq)]       
                    assert len(batch_predictions) == len(batch_references), "Predictions and  reference lists are different size"
                    if i==0:
                        beam1.append(batch_predictions)
                    elif i==1:
                        beam2.append(batch_predictions)
                    elif i==2:
                        beam3.append(batch_predictions)
                """
                for i in range(len(outputs)):
                    batch_predictions=tokenizer.batch_decode(outputs[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_references =tokenizer.batch_decode(test_instance['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_predictions = [batch_predictions[i:i + training_args.num_of_return_seq] for i in range(0, len(batch_predictions), training_args.num_of_return_seq)]       
                    assert len(batch_predictions) == len(batch_references), "Predictions and  reference lists are different size"
                    
                    for current_idex, (ref, pred, content) in enumerate(zip(batch_references, batch_predictions, test_instance['content'])):

                        f_gen.write(json.dumps({"instance_id": test_idx*training_args.test_batch_size + (current_idex +1),  "content": content, "reference": ref, "predictions":pred}, ensure_ascii=False) + "\n")

        f_gen.close()

if __name__ == '__main__':
    main()



