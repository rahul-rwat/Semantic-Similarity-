from sentence_transformers import SentenceTransformer, util
import numpy as np
import streamlit as st

#Header
st.header('Semantic Texual Similarity Model  ')
st.header('use the input boxes to insert the text and press enter')

#semantic similarity model 
model=SentenceTransformer('all-MiniLM-L6-v2')

#input 
sentence1=st.text_input("input1:")
sentence2=st.text_input("input2:")

# compute similarity scores of two embeddings
embedding1=model.encode(sentence1,convert_to_tensor=True)
embedding2=model.encode(sentence2,convert_to_tensor=True)

cosine_scores=util.pytorch_cos_sim(embedding1, embedding2)

st.write("text 1:", sentence1)
st.write("text 2:", sentence2)
st.write("similarity score:", cosine_scores.item())
