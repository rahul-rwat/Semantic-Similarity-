from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
import pandas as pd 

#semantic similarity model 
model=SentenceTransformer('stsb-roberta-large')


        
data1=pd.read_csv('Precily_Text_Similarity.csv',usecols=['text1'])
df1=(pd.DataFrame(data1))
data2=pd.read_csv('Precily_Text_Similarity.csv',usecols=['text2'])
df2=pd.DataFrame(data2)

df1.set_index("text1", inplace = True)
df2.set_index("text2", inplace = True)


x=range(1,5)
for x in range(1,5):
    sentence1=(df1.iloc[:, [0,5]])
print("text 1:",sentence1)

for x in range(1,5):
    sentence2=(df2.iloc[:, [0,5]])
print("text 2:",sentence2)


# compute similarity scores of two embeddings
embedding1=model.encode(sentence1,convert_to_tensor=True)
embedding2=model.encode(sentence2,convert_to_tensor=True)

cosine_scores=util.pytorch_cos_sim(embedding1, embedding2)

print("text 1:", sentence1)
print("text 2:", sentence2)
print("similarity score:", cosine_scores.item())
