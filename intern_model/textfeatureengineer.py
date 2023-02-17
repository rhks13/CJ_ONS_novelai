from arguments import Arguments
from transformers import AutoTokenizer
import torch
import pickle
import pandas as pd
import regex as re
import json
from sentence_transformers import SentenceTransformer

class TextFeatureEngineer():
    def __init__(self):
        super().__init__()
        self.eos = '==='

    def filter_data_in_genre(self, df, genre_list):
        df = df[df['genre'].apply(lambda x: x in genre_list)]
        return df

    def load_dataset(self, df_path):
        df = pd.read_csv(df_path, delimiter=',')
        return df
    def isNaN(self,string):
        return string != string
    def preprocess(self, df):
        prompt_list = []
        story_list = []
        whole_list = []
        for idx, row in df.iterrows():
            main_title = row['main_title']
            title = row['title']
            author = row['author']
            genre = row['genre']
            story = row['story']
            novel_id = row['novel_id']
            character= row['character_list']
            #if self.isNaN(character):
            #    character=''
            prompt = '제목:{}, 장르: {}, 등장인물: {}'.format(title, genre, character)
            #####장르 : ['판타지','로맨스','자유장르', '미스터리', '라이트노벨', '무협', '로판', '현판']####
            if len(story)>250 and genre=='판타지' and not self.isNaN(character):
                #print(story)
                preprocess_story = ' '.join(re.compile('[가-힣|a-z|A-Z|1-9|!?.~“”]+').findall(story)).replace('..','.').replace('~~','~')
                prompt_list.append(prompt)
                story_list.append(preprocess_story)
                if preprocess_story not in whole_list:    
                    #whole_list.append(preprocess_story)
                    whole_list.append(prompt+', 소설 내용:'+preprocess_story)
            else:
                continue
        #df = pd.DataFrame({'prompt': prompt_list,
        #           'story': story_list})
        df = pd.DataFrame({'text': whole_list})
        return df
        

class WordTokenizer():
    def __init__(self, model_name_or_path = None):
        super().__init__()
        tokenizer_args = Arguments("").tokenizer_args
        if model_name_or_path:
            tokenizer_args['pretrained_model_name_or_path'] = model_name_or_path
        self.tokenizer = self.load_tokenizer(tokenizer_args)

    def load_tokenizer(self, tokenizer_args):
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
        return tokenizer
    
    def _tokenizing(self, example):
        output = self.tokenizer(example)
        return output
    
    def tokenizing(self, examples):
        output = self.tokenizer(examples)
        return output
    
    def tokenizing_df(self, df):
        output = self.tokenizer(df['text'])
        return output
    
    
class SentenceTokenizer():
    def __init__(self, device='cuda'):
        super().__init__()
        if torch.cuda.is_available():
            self.device = device
        else:
            print("Unable to locate cuda device.. load model with cpu")
            self.device = 'cpu'
        self.key_name = 'text_embedder'
        self.embedder = self.load_tokenizer()

    def load_tokenizer(self, model_path = "jhgan/ko-sroberta-multitask"):
        model = SentenceTransformer(model_path,device=self.device)
        return model
    
    def tokenizing(self, examples):
        embedding_vectors = self.embedder.encode(examples, convert_to_tensor=True)
        return embedding_vectors
    
    def load_pickle(self, filepath):
        open_file = open(filepath, "rb")
        file = pickle.load(open_file)
        open_file.close()
        return file
    
    def load_embedding_vectors(self, filepath, method = 'sentence'):
        embedding_vectors = self.load_pickle(filepath)
        return embedding_vectors
    
    def get_separator(self, method):
        """method에 맞는 separator return
        """
        if method == 'sentence':
            separator = '\n'
        elif method == 'paragraph':
            separator = '\n\n'
        else:
            separator = None
        return separator

    def _split_text(self, example, method):
        separator = self.get_separator(method)
        example_splited = example.split(separator)
        return example_splited
    
    def split_text(self, df, method):
        """ 텍스트를 문장/문단 단위로 분할
        """

        splited_dict = dict()
        idx = 0
        splited_list = []
        for i in range(df.shape[0]):
            sample = df.iloc[i]
            text_splited = self._split_text(sample['text'], method=method)
            for text in text_splited:
                splited_dict[idx] = {
                    method: text,
                    'title': sample['title']}
                idx+=1
                splited_list.append(text)
        return splited_dict, splited_list
    
    def save_tokenizer(self):
        self.embedder.save('embedder') #FIXME: get_local_path 활용

    def save_embedding_vectors(self, embedding_vectors, method):
        filename = f'embedding_vectors_{method}.pickle'
    
    def _run(self, example, method = 'sentence'):
        splited_text = self._split_text(example, method=method)
        embedding_vector = self.tokenizing(splited_text)
        return splited_text, embedding_vector

    def run(self, df, method = 'sentence'):
        splited_dict, splited_list = self.split_text(df, method=method)
        embedding_vectors = self.tokenizing(splited_list)
        self.save_embedding_vectors(embedding_vectors, method=method)
        self.save_morphs(splited_dict, method)
        self.save_tokenizer()
        return splited_list, embedding_vectors
    
    def save_morphs(self, splited_dict, method):
        filename = f'morphs_{method}.json'
        with open(filename, 'w') as outfile:
            json.dump(splited_dict, outfile)

    def load_morphs(self, local, method):
        with open(local, "r") as json_file:
            json_data = json.load(json_file)
        return json_data
