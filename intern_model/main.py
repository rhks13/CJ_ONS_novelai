import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
from argparse import ArgumentParser
import dataloader 
import pandas as pd
import tqdm
import pdb
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def languagemodel_config(device):
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
  model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b").to(device)
  # if tokenizer.pad_token_id is None:
  #   tokenizer.add_special_tokens({'pad_token_id': '[PAD]'})
  # model.resize_token_embeddings(len(tokenizer))
  #attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in input_ids])
  return tokenizer, model

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  
  
  
  
def train_runner(args, train_dataloader, model, tokenizer):
  loss = 0
  #for _ in tqdm(range(args.epoch)):
  for _ in tqdm(range(1)):
    for (idx, prompt, story) in enumerate(train_dataloader):
      print(story, prompt, story)
      break
      
      
#   return 0
# def valid_runner(args, train_dataloader, model, tokenizer):
#   for _ in tqdm(range(args.epoch)):
    
#   return 0




#generation_data = pd.read_csv('./novel_data.csv', 'r', delimiter=',')
#train_data  = generation_data[:len(generation_data)*0.8]
#valid_data = generation_data[len(generation_data)*0.8:]
if __name__ == "__main__":
  set_seed(3)
  parser = ArgumentParser()
  parser.add_argument("--gpt3_config", type=str, default="ds_config.json")
  parser.add_argument("--epoch", default=50, type=int)
  parser.add_argument("--batch_size", default=128, type=int)
  parser.add_argument('--text_file', type=str, default='./data/novel_data_with_character_v2.csv')
  args = parser.parse_args()
  device = torch.device('cuda:0')
  tokenizer, model = languagemodel_config(device)
  text_file = pd.read_csv(args.text_file, delimiter='\t',encoding='utf-8')
  #pdb.set_trace()
  train_text_file = text_file[:int(len(text_file)*0.8)]
  valid_text_file = text_file[int(len(text_file)*0.8):]
  train_dataset = dataloader.GenerationDataset(train_text_file,tokenizer=tokenizer, device=device)
  valid_dataset = dataloader.GenerationDataset(valid_text_file,tokenizer=tokenizer, device=device)
  #for i in train_dataset:
  #  print(i)
    
  train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
  test_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True)
  
  
  train_runner(args, train_dataloader, model, tokenizer)
  
  
  
  
  
  prompt = '제목: 우리는, 등장인물: 혜선, 지호, 장르: 로맨스'
  # print(tokenizer.pad_token_id)
  # print(tokenizer)
  # with torch.no_grad():
  #   tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
  #   gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=300)
  #   generated = tokenizer.batch_decode(gen_tokens)[0]
    
  # print(generated)