import csv
import re
import pandas as pd
count = 0
def isNaN(string):
    return string != string
with open('/content/drive/MyDrive/CJ ONs/prac_pp.csv','w',encoding='utf-8') as csvfile:
    fieldnames = ['main_title','title','author','genre','story']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    with open('/content/drive/MyDrive/CJ ONs/prac.csv','r',encoding='utf-8') as spamreader:
        spamreader = pd.read_csv(spamreader, delimiter=',')
        for idx, row in spamreader.iterrows():
            main_title = row['main_title']
            title = row['title']
            author = row['author']
            genre = row['genre']
            story = row['story']
  
            # if isNaN(charater):
            #     charater=''
                
        
            # writer.writerow({'main_title': main_title ,'title': title,'author': author,'genre': genre,'story':story,'novel_id':novel_id,'character_list':charater})
            writer.writerow({'main_title': main_title ,'title': title,'author': author,'genre': genre,'story':story})
            count+=1
            if count>100:
                break