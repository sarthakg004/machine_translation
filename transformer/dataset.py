import torch
import torch.nn as nn
from torch.utils.data import  Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")],dtype=torch.int64)
        
        def __len__(self):
            return len(self.ds)
        
        def __getitem__(self, idx):
            src_target_pair = self.ds[idx]
            src_text = src_target_pair['translation'][self.src_lang]
            tgt_text = src_target_pair['translation'][self.tgt_lang]
            
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids  
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
            
            ## -2 for sos and eos tokens
            enc_num_padddings_tokens = self.seq_len - len(enc_input_tokens) - 2 
            ## -1 only for sos token during training 
            dec_num_padddings_tokens = self.seq_len - len(dec_input_tokens) - 1  
            
            if enc_num_padddings_tokens <0 or dec_num_padddings_tokens < 0:
                raise ValueError("seq_len is too long")
            
            ## Add sos and eos tokens to the source text
            encoder_input = torch.cat([self.sos_token,
                                       torch.Tensor(enc_input_tokens, dtype=torch.int64),
                                       self.eos_token,
                                       torch.Tensor([self.pad_token]*enc_num_padddings_tokens, dtype=torch.int64)])
            
            # Add SOS to the decoder input
            decoder_input  = torch.cat([self.sos_token,
                                       torch.Tensor(dec_input_tokens, dtype=torch.int64),
                                       torch.Tensor([self.pad_token]*dec_num_padddings_tokens, dtype=torch.int64)])

            # ADD EOS to the label 
            label = torch.cat([torch.Tensor(dec_input_tokens, dtype=torch.int64),
                              self.eos_token,
                              torch.Tensor([self.pad_token]*dec_num_padddings_tokens, dtype=torch.int64)])
            
            assert encoder_input.size[0] == self.seq_len
            assert decoder_input.size[0] == self.seq_len
            assert label.size[0] == self.seq_len
            
            return{
                'encoder_input': encoder_input, # Seq len
                'decoder_input': decoder_input, # Seq len
                'encoder_mask' : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
                'decoder_mask' : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1,seq_len) & (1,seq_len, seq_len)
                'label': label, # Seq len
                'src_text': src_text,
                'tgt_text': tgt_text
            }
        
        
def casual_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask ==0
            