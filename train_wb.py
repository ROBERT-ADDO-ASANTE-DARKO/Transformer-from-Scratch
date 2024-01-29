from requests import get
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
           
    # Evaluate the charcter error rate
    # Compute the char error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})
    
    # Compute the word error rate
    metric = torch.metrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})
    
    # Compute the BLEU metric
    metric = torch.metrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})
    
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        # Create a Tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
        
        #   We will use the `ByteLevelBPETokenizer` which is fast to train and lightweight but also quite efficient on CPU.
        # Subword tokenization allows us to represent words at the character level by  
        # splitting them into smaller parts (characters). For instance, the word "hello" can be represented as "hel<|>lo".
        # splitting them into smaller parts (characters). For instance, the word "hello" can be represented as ["hel", "lo"] or ["he", "ll", "o"].
        # splitting them into smaller parts (characters). For example, the word "hello" can be represented as ["hel", "lo"].
        # splitting them into smaller parts (characters). For example, the word "hello" can be split into "hel", "lo".
        # splitting them into smaller parts (characters). For example, the word "hello" can be represented as "hel##lo".
        # splitting them into smaller parts (characters or subwords). For example, the    
        # word "hello" can be represented as ["hel", "lo"] which are two separate            
        # characters.                                                                                                                                        
        # The ByteLevelBPE tokenizer was originally proposed in this paper:                          
        # https://arxiv.org/abs/1609.08753                                                                                                                  
        # Here we set the vocab size to 40000. It means that the tokenizer will learn          
        # a vocabulary of 40000 different characters during training. If your text contains          
        # many unique characters, you may want to increase this number.                              You should then retrain      
        # many unique characters, you might want to increase this number. In practice,                          
        # a higher value than 40000 usually does not provide much improvement. You should          adjust            
        # this parameter according to your specific task.                                              You should adjust it accordingly.         
        # many unique characters, you might want to increase this number. In practice,                          
        # a higher value than 40000 leads to better results but requires more computational resources.
        
        #yield [item['src_text'][lang], item['tgt_text'][lang]]
        
def get_ds(config):
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Find the maximum length of each sentence in the source and target sentences
    max_len_src = 0
    max_len_tgt = 0
    
    #for batch in train_ds_raw.shuffle().batch(1024):
    #    src_lens = [len([x for x in ex if x != tokenizer_src.eos_token]) for ex in batch["translation"]]
    #    tgt_lens = [len([x for x in ex if x != tokenizer_tgt.eos_token]) for ex in batch["translation_src"]]
    #    max_len_src = max(max_len_src, max(src_lens), max(tgt_lns for tgt_lns in zip(src_lens, tgts)))
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config['d_model'])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state
        
        #def loss_fn(tokenizer_src):
        #    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
        #    return loss_fn
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # define our custom x-axis metric
    wandb.define_metric("global_step")
    # define which metric will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)            
            decoder_mask = batch['decoder_mask'].to(device)
            
            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)
            
            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            
        # Run the validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)
        
        # Save the model at the end of every epoch
        model.filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config["num_epochs"] = 30
    config['preload'] = None
    
    wandb.init(
        # Set the wandb project where this run will be logged
        project="pytorch-transformer",
        
        # Track hyperparameters and run metadata
        config=config
    )
    train_model(config)
# Helper function to save a list of sentences (one per line) to file
#save_to_file = lambda x, fname : [print(*i.split('\n'), sep='\n', file=f) for i in x for f in fname]

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