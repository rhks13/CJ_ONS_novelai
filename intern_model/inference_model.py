import re
from textfeatureengineer import TextFeatureEngineer, WordTokenizer
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM 
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import pdb
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import pickle
from os.path import exists
import torch.nn as nn




font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
#print(mpl.font_manager.fontManager.ttflist)
#print("font",font)
rc('font', family=font)
class TextModelInferencer():
    def __init__(self, model_path, key_name = "generation_model_1", device='cuda'):
        super().__init__()
        self.text_feature_engineer = TextFeatureEngineer()
        #self.preposition = self.add_preposition()
        
        self.bucket_type = 'model'
        self.key_name = key_name
        
        if torch.cuda.is_available():
            self.device = device
        else:
            print("Unable to locate cuda device.. load model with cpu")
            self.device = 'cpu'

            
        if model_path is not None:
            self.model, self.word_tokenizer = self.load_model(model_path)
        else:
            print('Unable to find model.. terminate program')
    

    def load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True).to(device=self.device, non_blocking=True)
        word_tokenizer = WordTokenizer(model_name_or_path = model_path)
        return model, word_tokenizer
    
    def switch_device(self, device): 
        if self.device == device:
            pass
        else:
            print(f'Change device from {self.device} to {device}')
            self.model = self.model.to(device=device, non_blocking=True)
            self.device = device
    
    def add_preposition(self, datapath = 'preposition.txt'):
        f = open(datapath, 'r', encoding='utf-8')
        preposition = f.readlines()
        f.close()
        preposition = [p.strip() for p in preposition]
        return preposition

    def get_valid_word_list(self, word_list):
        """
        word_list가 공백일 시 제거
        """

        stripped_bad_words = []
        for word in word_list:
            word = word.strip()
            if len(word) == 0:
                continue
            else:
                stripped_bad_words.append(word)
        return stripped_bad_words


    def get_id_comb(self, bad_word):
        """
        bad_word에 가능한 조합 생성하여 token id로 변환
        """
        start_word = '▁'
        vocab_list = self.word_tokenizer.tokenizer.vocab.keys()

        comb = [bad_word[:i] for i in range(1, len(bad_word)+1)]
        comb = [start_word + word for word in comb] + comb
        
        bad_word_with_preposition = [bad_word + p for p in self.preposition]
        bad_word_with_preposition = [start_word + word for word in bad_word_with_preposition] + bad_word_with_preposition

        tokens_cand = comb + bad_word_with_preposition

        ids_list = []
        for word in tokens_cand:
            if word in vocab_list:
                ids_list.append([self.word_tokenizer.tokenizer.vocab[word]])
        return ids_list

    def parse_generated_text(self, text, print_step):
        if '.' in text:
            splited = text.split('.')
            splited = [text.strip() + '.\n' for text in splited]
            return ''.join(splited)
        else:
            splited = text.split(' ')
            for i, s in enumerate(splited):
                if i % print_step == (print_step - 1):
                    splited[i] = s.strip() + '\n'
            return ' '.join(splited).strip()
    
    def remove_unknown_weird_token(self, text):
        return text.replace('<unk>','').replace('※','')

    def parse_eos(self, text):
        text = text.split(self.text_feature_engineer.eos)[0]
        text = text.strip()
        return text
    
    def postprocessing(self, prompt, generated):
        # generated = self.parse_eos(generated)
        # if reline:
        #     generated = self.parse_generated_text(generated, print_step)
        generated = generated[len(prompt):].split('===')[0]
        generated = self.remove_unknown_weird_token(generated)
        return generated
    
    def inference(self, prompt, temperature=0.8, top_p=0.8, max_length=64,repetition_penalty=1.2,top_k=50, no_repeat_ngram_size=3, seed=0, bad_words=[], reline = False):
        """_summary_

        Args:
            prompt (string): 키워드로 입력할 텍스트
            temperature (float, optional, defaults to 1.0) — The value used to module the next token probabilities.
            top_p (float, optional):  If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.. Defaults to 1.0.
            top_k (int, optional):The number of highest probability vocabulary tokens to keep for top-k-filtering.  defaults to 50
            max_length (int, optional): max token 개수. Defaults to 128.
            no_repeat_ngram_size (int, optional): 같은 생성 결과가 나올 수 있는 최대 ngram. Defaults to 3.
            seed (int, optional): random seed. Defaults to 0.
            bad_words (list, optional): 생성 결과에서 제외할 단어. Defaults to [].
            encoder_no_repeat_ngram_size (int, optional, defaults to 0) — If set to int > 0, all ngrams of that size that occur in the encoder_input_ids cannot occur in the decoder_input_ids.
            repetition_penalty (float, optional, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
            length_penalty (float, optional, defaults to 1.0) — Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent 
            to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
        Returns:
            _type_: _description_
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        bad_words = [ '...', '....', '(중략)','http']
        nlls = []
        #print("model config", self.model.config)
        stride = 512
        prev_end_loc = 0
        seq_len = max_length
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc + prev_end_loc
            #input_ids = tokens.input_ids[:, begin_loc:end_loc].to(self.device)

            with torch.no_grad():
                tokens = self.word_tokenizer.tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
                gen_tokens = self.model.generate(tokens, do_sample=True, temperature=temperature, max_new_tokens=max_length, top_k=top_k, top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size, 
                                                repetition_penalty=repetition_penalty,bad_words_ids=[self.word_tokenizer.tokenizer.encode(bad_word) for bad_word in bad_words],
                                                eos_token_id=self.word_tokenizer.tokenizer.eos_token_id,pad_token_id=self.word_tokenizer.tokenizer.pad_token_id)
                target_ids = tokens.clone()
                target_ids[:, :-trg_len] = -100
                """hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is 
                passed or when config.output_hidden_states=True) — 
                Tuple of torch.FloatTensor (one for the output of  the embeddings + 
                one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)."""
                outputs_embed = self.model(input_ids= tokens, labels=target_ids)
                ##embed original and gen####
                gen_embed = self.model(input_ids=gen_tokens)
                generated = self.word_tokenizer.tokenizer.batch_decode(gen_tokens)[0]
                ############################
                #print(len(self.model(input_ids= tokens)))
                #print(len(self.model(input_ids=gen_tokens)))
                outputs_embed_hidden = torch.squeeze(outputs_embed.hidden_states[1])[-1:]
                gen_embed_hidden = torch.squeeze(gen_embed.hidden_states[1])[-1:]
                neg_log_likelihood = outputs_embed.loss * trg_len
                nlls.append(neg_log_likelihood)
            generated = self.postprocessing(prompt, generated)
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        print("perplexity of this tuned language model is {}".format(ppl))
        return generated, outputs_embed_hidden, gen_embed_hidden, ppl
    
    
    
class polyglotGen():
    def __init__(self,  model_name, key_name = "generation_model_1", device='cuda'):
        self.key_name = key_name
        
        if torch.cuda.is_available():
            self.device = device
        else:
            print("Unable to locate cuda device.. load model with cpu")
            self.device = 'cpu'
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, pad_token_id=self.tokenizer.eos_token_id, torch_dtype='auto',  output_hidden_states=True, low_cpu_mem_usage=True).to(device='cuda', non_blocking=True)
    
    def remove_unknown_weird_token(self, text):
        return text.replace('<unk>','').replace('※','')

    def parse_eos(self, text):
        text = text.split(self.text_feature_engineer.eos)[0]
        text = text.strip()
        return text
    
    def postprocessing(self, prompt, generated):
        # generated = self.parse_eos(generated)
        # if reline:
        #     generated = self.parse_generated_text(generated, print_step)
        generated = generated[len(prompt):].split('===')[0]
        generated = self.remove_unknown_weird_token(generated)
        return generated
    def inference(self, prompt, temperature=0.8, max_length=64,top_k=50, top_p=0.8, no_repeat_ngram_size=3,repetition_penalty=1.2, seed=0):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        bad_words = [ '...', '....', '(중략)','http']
        nlls = []
        #print("model config", self.model.config)
        stride = 512
        prev_end_loc = 0
        seq_len = max_length
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc + prev_end_loc
            #input_ids = tokens.input_ids[:, begin_loc:end_loc].to(self.device)

            with torch.no_grad():
                tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
                gen_tokens = self.model.generate(tokens, do_sample=True, temperature=temperature, max_new_tokens=max_length, top_k=top_k, top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size, 
                                                repetition_penalty=repetition_penalty,bad_words_ids=[self.tokenizer.encode(bad_word) for bad_word in bad_words],
                                                eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                target_ids = tokens.clone()
                target_ids[:, :-trg_len] = -100
                """hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is 
                passed or when config.output_hidden_states=True) — 
                Tuple of torch.FloatTensor (one for the output of  the embeddings + 
                one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)."""
                outputs_embed = self.model(input_ids= tokens, labels=target_ids)
                ##embed original and gen####
                gen_embed = self.model(input_ids=gen_tokens)
                generated = self.tokenizer.batch_decode(gen_tokens)[0]
                ############################
                #print(len(self.model(input_ids= tokens)))
                #print(len(self.model(input_ids=gen_tokens)))
                outputs_embed_hidden = torch.squeeze(outputs_embed.hidden_states[1])[-1:]
                gen_embed_hidden = torch.squeeze(gen_embed.hidden_states[1])[-1:]
                neg_log_likelihood = outputs_embed.loss * trg_len
                nlls.append(neg_log_likelihood)
            generated = self.postprocessing(prompt, generated)
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        print("perplexity of this untuned language model is {}".format(ppl))
        return generated, outputs_embed_hidden, gen_embed_hidden, ppl
    

class sbert_embedding():
    def __init__(self, device='cuda'):
        self.device = device
        self.model = SentenceTransformer('smartmind/ko-sbert-augSTS-maxlength512')
        self.tokenizer = AutoTokenizer.from_pretrained('smartmind/ko-sbert-augSTS-maxlength512')
        
    # def mean_pooling(self, model_output, attention_mask):
    #     #pdb.set_trace()
    #     sentence = model_output.sentence_embedding #First element of model_output contains all token embeddings
    #     #input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
    #     #return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)   
    
    def embed(self, prompt, generated):
            #### read#####
            #output of hiddden for GPT is tuple: one for output of the embedding, one for embedding + output of each layer) 
        
        with torch.no_grad():
            prompt_encoded = self.tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
            generated_encoded = self.tokenizer(generated, padding=True, truncation=True, return_tensors='pt')
            # prompt_model_output = self.model(**prompt_encoded).to(self.device)
            # generated_model_output = self.model(**generated_encoded).to(self.device)
            prompt_model_output = self.model(prompt_encoded).sentence_embedding.cpu().detach().numpy()
            generated_model_output = self.model(generated_encoded).sentence_embedding.cpu().detach().numpy()
        #prompt_sentence_embeddings = self.mean_pooling(prompt_model_output, prompt_encoded['attention_mask'])
        #generated_sentence_embeddings = self.mean_pooling(generated_model_output, generated_encoded['attention_mask'])

        return prompt_model_output, generated_model_output
def isNaN(string):
    return string != string
    
def postprocessing(text):
    return 0
if __name__ == '__main__':
    
    parser = ArgumentParser("학습")
    parser.add_argument("--model_path", help="model_path", default = './model_fantasy/checkpoint-3100', type=str)
    parser.add_argument("--model_key_name", help="model_key_name", default = 'generation', type=str)
    parser.add_argument("--max_length", help="max_length", default = '128', type=int)
    args = parser.parse_args()
    ours = TextModelInferencer(key_name = args.model_key_name, model_path=args.model_path)
    model_name  = 'EleutherAI/polyglot-ko-1.3b'
    poly_model = polyglotGen(model_name=model_name,key_name = args.model_key_name)
    pdb.set_trace()
    pickle_path = './feature_Save/feature_save.pickle'
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    tsne_arr=[]
    genre_arr = []
    tsne_model = TSNE(n_components=2, perplexity = 30,random_state=123)
    if exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            
            data = pickle.load(f)
        #tsne_arr.extend(data['prompt_feature'])
        tsne_arr.extend(data['generated_feature'])
        tsne_arr = np.array(tsne_arr)
        genre_arr.extend(data['genre'])
    else:
        valid_data = pd.read_csv('./data/novel_data_for_embedding_space.csv', delimiter=',')

        unique_genre_arr = []
        genre_arr = []
        prompt_embed_arr = []
        generated_embed_arr = []
        prompt_arr = []
        original_arr = []
        generated_arr = []
        ppl_arr = []
        #pdb.set_trace()
        for idx, row in valid_data.iterrows():
            genre = row['genre']
            title = row['title']
            if genre not in unique_genre_arr:
                unique_genre_arr.append(genre)
            genre_arr.append(genre)
            character = row['character_list']
            if isNaN(character):
                characeter=''
            prompt = """제목: {}
            '장르: {}
            등장인물: {}
            """.format('우리는','판타지', '아르곤, 제니스')
            #pdb.set_trace()
            generated_decoded, prompt_embedding, gen_embedding, ppl = ours.inference(prompt, max_length = args.max_length)
            original, _, _, _ = poly_model.inference(prompt, max_length = args.max_length)
            prompt_arr.append(prompt)
            generated_arr.append(generated_decoded)
            original_arr.append(original)
            ppl_arr.append(ppl)
            print("Original:",original )
            print("-"*20)
            print("Ours:",generated_decoded)
            ####
            
            prompt_embed_arr.append(prompt_embedding.detach().cpu().numpy())
            generated_embed_arr.append(gen_embedding.detach().cpu().numpy())
            tsne_arr.extend(gen_embedding.detach().cpu().numpy())
            # embed = sbert_embedding() 
            # prompt_embedded, generated_embedded = embed.embed(original, generated_decoded)
            # prompt_embed_arr.append(prompt_embedded)
            # generated_embed_arr.append(generated_embedded)
            # tsne_arr.extend(prompt_embedded)
            # tsne_arr.extend(generated_embedded)
            ##
        pdb.set_trace()
        tsne_arr = np.array(tsne_arr)
        feature_save = {'prompt':prompt_arr,
                        'generated':generated_arr,
                        'original': original_arr,
            'prompt_feature':prompt_embed_arr,
                'generated_feature':generated_embed_arr,
                'genre':genre_arr,
                'perplexity':ppl_arr}
        with open(pickle_path, 'wb') as f:
            pickle.dump(feature_save,f ,protocol=pickle.HIGHEST_PROTOCOL)
    tsne_arr = np.squeeze(tsne_arr)
    transformed = tsne_model.fit_transform(tsne_arr)

    #print(transformed.shape)
    #pdb.set_trace()
    PD_TSNE = pd.DataFrame(transformed)
    PD_TSNE['cluster'] = genre_arr
    PD_TSNE.columns = ['x1','x2','cluster']
    embedding_plot = sns.scatterplot(data=PD_TSNE, x='x1', y='x2',hue='cluster')
    fig = embedding_plot.get_figure()
    fig.savefig('out_1.png')
    



    with open('./result/novel_created.csv','w', encoding='utf-8') as csvfile:
        fieldnames = ['Original', 'Generated']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Original': prompt_arr ,'Generated': generated_arr})