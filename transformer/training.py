import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # ds_raw = load_dataset("cfilt/iitb-english-hindi")
    ds_raw = load_dataset('opus_books',f'{config['lang_src']}-{config['lang_tgt']}',split = 'train')
    
    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])
    
    train_ds_size = int(0.8*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size 
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print('Max length of source text:', max_len_src)
    print('Max length of target text:', max_len_tgt)
    
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = False)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
        
    
def get_model(config, vocab_src_size, vocab_tgt_size):
    model = build_transformer(vocab_src_size, vocab_tgt_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

    
    
    