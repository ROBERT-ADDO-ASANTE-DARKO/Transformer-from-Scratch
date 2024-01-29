from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import torch.functional as F

import time
import numpy as np

import warnings
from tqdm import tqdm
import os
from pathlib import Path

#HuggingFace datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb

import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id(['SOS'])
    eos_idx = tokenizer_tgt.token_to_id(['EOS'])
    
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    try:
        # Get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
        
    with torch.no_grad():
        for batch in validation_ds:
           count += 1
           encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
           encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
           
           # Check that the batch size is 1
           assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
           
           model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
           
           source_text = batch["src_text"][0]
           target_text = batch["tgt_text"][0]
           model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
           
           source_texts.append(source_text)
           expected.append(target_text)
           predicted.append(model_out_text)
           
           # Print the source, target and model output
           print_msg('-'*console_width)
           print_msg(f"{f'SOURCE: ':>12}{source_text}")
           print_msg(f"{f'TARGET: ':>12}{target_text}")
           print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
           
           if count == num_examples:
               print_msg('-'*console_width)
               break
            
#def beam_search(model, source, source_mask, tokenizer_src, tokenizer_tgt, num_beams, max_length, device):
#    batch_size = source.shape[0]
#    vocab_size = len(tokenizer_tgt)
    
#    def expand_labels(labels, beam_index):
#        expanded_labels = labels + np.zeros((labels.shape[0], num_beams-1), dtype=labels.dtype)
#        expanded_labels[:, beam_index] = 2
#        return expanded_labels
    
    # Prepare inputs
#    sos_idx = tokenizer_tgt.token_to_id(['SOS'])
#    eos_idx = tokenizer_tgt.token_to_id(['EOS'])
#    source = source.unsqueeze(1).repeat([1,num_beams]+list(source.shape[1:]))
#    source_mask = source_mask.unsqueeze(1).repeat([1,num_beams]+list(source_mask.shape[1:]))
#    decoded = [np.array([sos_idx]*num_beams)]
    
    # Start timer
#    start = time.time()
    
    # Beam search loop
#    for _ in range(max_length):
#       encoder_output = model.encode(source)[0][:batch_size]
        
        # Stop if all sentences end
#        if np.all(decoded[-1][:,-1:]==eos_idx):
#            break
            
        # Get predictions from the model (beam search)
#        outputs = model.decode(encoder_output, source_mask, None, None)
#        logits = outputs[:, :, :-1, :vocab_size].expand(-1,-1,-1,num_beams*vocab_size)
#        logits = logits.reshape(logits.shape[0], logits.shape[1], logits.shape[2]*logits.shape[3])
#        probs = F.softmax(logits / np.sqrt(model.d_model),dim=-1)
#        next_words = torch.multinomial(probs, num_samples=num_beams)
#        next_words = next_words.view(batch_size, num_beams, -1)
        
        # Update the decoding results
#        decoded.append(next_words)
        
        # Check for the end of sentence and add to the result
#        masked_next_words = next_words * (1 - source_mask[...,None]).long()
#        indices = torch.argmax(masked_next_words, dim=-1)
#        positions = torch.utils.convert_tokens_to_ids(tokenizer_tgt.convert_tokens_to_strings(indices))
#        finished = (positions == eos_idx).any(-1)
#        if np.all(finished):
#            break
#    else:
#        print("WARNING: Beam search did not complete.")
    
    # Gather all beams together and remove padding
#    decoded = np.concatenate(decoded[1:], axis=-1)
#    decoded = decoded[(np.arange(len(decoded)),utils.get_last_occurrence(decoded != pad_idx,axis=-1))]
    
    # Return as strings
 #   return tokenizer_tgt.convert_ids_to_tokens(decoded.tolist())