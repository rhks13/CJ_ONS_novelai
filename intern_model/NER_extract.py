import csv
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from pororo import Pororo
import torch
import math


def isNaN(string):
    return string != string
with open('./data/novel_data_all_12_08_with_character_fantasy.csv','w',encoding='utf-8') as csvfile:
    fieldnames = ['main_title','title','author','genre','story','novel_id','character_list']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    novel_id = 0
    with open('data/novel_data_all_12_12_1.csv','r', newline='',encoding='utf-8') as spamreader:
        spamreader = pd.read_csv(spamreader, delimiter=',')
        count =0
        preprocessed_story_arr =[]
        for idx, row in spamreader.iterrows():
            count +=1
            if count > 17000:
                character_list = []
                main_title = row['main_title']
                title = row['title']
                author = row['author']
                genre = row['genre']
                story = row['story']
                novel_id = row['novel_id']
                preprocessed_story = ' '.join(re.compile('[가-힣|a-z|A-Z|!~?,.''“”():]+').findall(story))
                #### pororo
                preprocessed_story_arr.append(preprocessed_story)
                print(genre)
                
                if  isNaN(title): ## if title is nan
                    title = '막화'
                if 'e-book' in story or len(story) < 120  or genre !='판타지':
                    continue
                
                if len(preprocessed_story) > 512:
                    preprocessed_story = preprocessed_story[:512]
                ner = Pororo(task="ner", lang="ko")
                result = (ner(preprocessed_story))

                for i in result:
                    if i[1] == 'PERSON':
                        name = i[0].split()
                        #print(name)
                        name = name[0].rstrip().lstrip()
                        if name not in character_list:
                            character_list.append(name)
                #print(','.join(chraceter_list))
                writer.writerow({'main_title': main_title ,'title': title,'author': author,'genre': genre,'story':preprocessed_story,'novel_id':novel_id,'character_list':','.join(character_list)})
            #f_w.write(str(main_title)+'\t'+str(title)+'\t'+str(author)+'\t'+str(genre)+'\t'+str(story)+'\t'+str(novel_id)+'\t'+','.join(character_list)+'\n')
            ######
            else:
                continue
            
            # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")        
            # model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
            # classifier = pipeline("ner", model=model, tokenizer=tokenizer)
            # #print(preprocessed_story)
            # result = (classifier(preprocessed_story))
            # print(result)
            # result_id = 0
            # character_list = []
            # for i in result:
            #     if i['entity']=='I-PER' and result_id+1 == i['index']:
            #         before = character_list.pop()
            #         character_list.append(before+i['word'])
            #     elif i['entity']=='I-PER':
            #         result_id = i['index']
            #         character_list.append(i['word'])
            #         print(i['word'])
            # print(character_list)
            # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
            # model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
            # classifier = pipeline("ner", model=model, tokenizer=tokenizer)
            # result = classifier(preprocessed_story)
            # print(result)
            # if result['entity'] == 'I-PERSON':
            #     print(result['word'])
            # # hannanum_tagged = hannanum.pos(preprocessed_story)
            # twitter_tagged = twitter.pos(preprocessed_story)
            #kkma_tagged = kkma.pos(preprocessed_story)
            #komoran_tagged = (komoran.pos(preprocessed_story))
            
            
            # for i in kkma_tagged:
            #     if i[1] == 'NNP' and i[0] not in kkma_list:
            #         kkma_list.append(i)
            # for i in komoran_tagged:
            #     if i[1] == 'NNP' and i[0] not in komoran_list:
            #         komoran_list.append(i)
            # for i in kkma_list:
            #     result = classifier(i[0])
            #     if result['entity'] == 'I-PERSON':
            #         print(result['word'])
            # #print(komoran_list)
            # for i in kkma_tagged:
            #     if i[1] == 'NNP' and i[0] not in kkma_list:
            #         kkma_list.append(i)
            # for i in twitter_tagged:
            #     if i[1] == 'Noun' and i[0] not in twitter_list:
            #         twitter_list.append(i)
            # for i in hannanum_tagged:
            #     if i[1] == 'N' and i[0] not in twitter_list:
            #         hannaum_list.append(i)
            #for i in twitter_tagged:
            #    print(i)
            # for (a,b,c,d) in zip (komoran_list, kkma_list, twitter_list, hannaum_list):
            #     print(a[0]) 
            