import os
import glob
import csv
from re import S
from torch.utils import data
import tqdm
import numpy as np
import sys 

import torch
from torch.utils.data import Dataset

import transformers

def load_the_dataset(args, data_path, tokenizer, mode):
    files = [data_path]
    if os.path.isdir(data_path):
        files = glob(f'{data_path}/*.tsv', recursive=True)
    #if mode == 'test':
    #    print(files)
    #    datasets = [PrepareDataset(args, file, tokenizer, mode) for file in files]
    #    return datasets
    print(PrepareDataset(args, files, tokenizer, mode))
    return PrepareDataset(args, files, tokenizer, mode)

class PrepareDataset(Dataset):
    def __init__(self, args, files, tokenizer, mode):
        self.files = files
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length

        self.data = []
        if isinstance(self.files, list):
            for file in self.files:
                if args.read_n_data_obj != -1: 
                    self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                        count, i in enumerate(open(self.files[0], 'r', encoding='utf8')) if \
                            count < args.read_n_data_obj ), delimiter='\t', quoting=csv.QUOTE_NONE))
                else:
                    self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                        i in open(self.files[0], 'r', encoding='utf8')), delimiter='\t', quoting=csv.QUOTE_NONE))
        else:
            self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                        i in open(self.files, 'r', encoding='utf8')), delimiter='\t', quoting=csv.QUOTE_NONE))
        
        
    def __len__(self):
        return len(self.data)

    def create_input_instance(self, encoided_ids, max_sequences_length):
        encoided_ids +=[-100 for _ in range(max_sequences_length - len(encoided_ids))]
        encoided_ids = torch.LongTensor(encoided_ids)
        input_mask = (encoided_ids != -100).long()
        encoided_ids.masked_fill_(encoided_ids == -100, self.tokenizer.pad_token_id)
        return {
            'input_ids': encoided_ids,
            'attention_mask' : input_mask,
        }


    def features_bin(self, lines):
        if not isinstance(lines[0], list):
            examples = [lines]
        else:
            examples = lines
        data_instances = []
        
        for line in examples:
            session, label1, label2, label3 = line[0], line[1], line[2], line[3]
            #session = ",".join(session.split('||')[:-1])
            encoded = self.tokenizer.encode(session + self.tokenizer.eos_token, add_special_tokens=False)
            encoded = encoded[::-1][:self.max_source_length][::-1]
            data_instance = self.create_input_instance(encoded, self.max_source_length)

            encoded_label1 = self.tokenizer.encode(label1 + self.tokenizer.eos_token, add_special_tokens=False)
            encoded_label1 = encoded_label1[::-1][:self.max_target_length][::-1]
            data_instance_label1 = self.create_input_instance(encoded_label1, self.max_target_length)

            encoded_label2 = self.tokenizer.encode(label2 + self.tokenizer.eos_token, add_special_tokens=False)
            encoded_label2 = encoded_label2[::-1][:self.max_target_length][::-1]
            data_instance_label2 = self.create_input_instance(encoded_label2, self.max_target_length)

            encoded_label3 = self.tokenizer.encode(label3 + self.tokenizer.eos_token, add_special_tokens=False)
            encoded_label3 = encoded_label3[::-1][:self.max_target_length][::-1]
            data_instance_label3 = self.create_input_instance(encoded_label3, self.max_target_length)

            data_instance['labels'] = data_instance_label1['input_ids']
            data_instance['decoder_attention_mask'] = data_instance_label1['attention_mask']
            data_instance['labels1'] = data_instance_label2['input_ids']
            data_instance['decoder_attention_mask1'] = data_instance_label2['attention_mask']
            data_instance['labels2'] = data_instance_label3['input_ids']
            data_instance['decoder_attention_mask2'] = data_instance_label3['attention_mask']

            if self.mode == 'test':
                #data_instance['prefix'] = prefix.strip()
                data_instance['content'] = line[0]

            if not isinstance(lines[0], list):
                return data_instance
            else:
                data_instances.append(data_instance)
        #print(data_instances)
        return data_instances

    def set_epoch(self, epoch):
        self.random_state= np.random.RandomState(epoch)

    def __getitem__(self, indx):
        #print("index : ",indx)
        #print("data length : ",len(self.data))
        return self.features_bin(self.data[indx])
