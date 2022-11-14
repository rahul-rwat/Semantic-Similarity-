from sentence_transformers import SentenceTransformer, util
import numpy as np

#semantic similarity model 
model=SentenceTransformer('stsb-roberta-large')

#input 
sentence1=input("input1:")
sentence2=input("Input 2:")

# compute similarity scores of two embeddings
embedding1=model.encode(sentence1,convert_to_tensor=True)
embedding2=model.encode(sentence2,convert_to_tensor=True)

cosine_scores=util.pytorch_cos_sim(embedding1, embedding2)

print("text 1:", sentence1)
print("text 2:", sentence2)
print("similarity score:", cosine_scores.item())
