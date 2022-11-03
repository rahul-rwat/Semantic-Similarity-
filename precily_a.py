from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
import pandas as pd 

#semantic similarity model 
model=SentenceTransformer('stsb-roberta-large')

# Open file
#with open('Precily_Text_Similarity.csv') as file_obj:

    # Skips the heading
    #heading = next(file_obj)  
    
        #reader_obj = csv.DictReader(file_obj)

    # Iterate over each row in the csv file
    # using reader object
    #for row in reader_obj:
        
data1=pd.read_csv('Precily_Text_Similarity.csv',usecols=['text1'])
data2=pd.read_csv('Precily_Text_Similarity.csv',usecols=['text2'])
#data.text1


x=range(1,5)
for x in data1:
    sentence1=(data1[x])
    embedding1=model.encode(sentence1,convert_to_tensor=True)
for x in data2:
    sentence2=(data2[x])
    embedding2=model.encode(sentence2,convert_to_tensor=True)

    # compute similarity scores of two embeddings
    cosine_scores=util.pytorch_cos_sim(embedding1, embedding2)

print("text 1:", sentence1)
print("text 2:", sentence2)
print("similarity score:", cosine_scores.item())
