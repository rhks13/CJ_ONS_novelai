import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import csv
import re
import pdb

class GenerationDataset(Dataset):
    def __init__(self, csv_file,tokenizer, device):
        super(GenerationDataset, self).__init__()
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.device = device
        #self.main_title, self.title, self.author, self.genre, self.preprocess_story, self.novel_id  = self.preprocess(self.csv_file, self.tokenizer, self.device)
        self.prompt, self.story = self.preprocess(self.csv_file, self.tokenizer, self.device)
        self.len = len(self.main_title)
    def preprocess(self, pd_data, tokenizer, device):
        ###
        #'main_title' ,'title','author','genre','story', novel_id
        ##
        #main_title_list, title_list, author_list, genre_list, story_list, novel_id_list, character_list = [],[],[],[],[],[],[]
        prompt_list, story_list = [], []
        for idx, row in pd_data.iterrows():
            pdb.set_trace()
            main_title = row['main_title']
            title = row['title']
            author = row['author']
            genre = row['genre']
            story = row['story']
            novel_id = row['novel_id']
            character= row['characeter_list'] ## to be added
            
            if 'e-book' in story or title=='title':
                continue
            preprocess_story = ' '.join(re.compile('[가-힣|a-z|A-Z|!?.“”]+').findall(story))
            #main_title_list.append(main_title)
            #title_list.append(title)
            #author_list.append(author)
            #genre_list.append(genre)
            #story_list.append(preprocess_story)
            #novel_id_list.append(novel_id)
           #character_list.append(character)
           
          ###tokenize
            ####
            ## string to concat genre, characters, title
            prompt = '제목:{}, 장르: {}, 등장인물: {}'.format(title, genre, character)
            token_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
            token_story = tokenizer.encode(preprocess_story, return_tensors='pt').to(device=device, non_blocking=True)
            ###
        #pdb.set_trace()
        #return main_title, title, author, genre, preprocess_story, novel_id 
        return prompt_list, story_list ##character_list
    
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        return self.prompt[idx], self.story[idx]
        #return self.title[idx], self.genre[idx], self.preprocess_story[idx] ## to be added character_list[idx]
        