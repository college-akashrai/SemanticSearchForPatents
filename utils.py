# -*- coding: utf-8 -*-
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import pandas as pd
import numpy as np
from datasets.arrow_dataset import Dataset
import faiss
# from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import AutoTokenizer, TFAutoModel
import streamlit as st
# model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    # text=str(text_list)
    # prompt="Give a generic information on the subject under 80 words."
    # # skip_len=len(prompt)
    # prompt+=text
    # llm_tokenize,llm_model,tokenizer,model=load_model()
    _1,_2,tokenizer,model=load_model()
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    # generate_text(prompt,llm_model,llm_tokenize)
    return cls_pooling(model_output)

def semantics(query):

    # Load the dataset from the saved directory
    loaded_dataset = Dataset.load_from_disk('embeddings_dataset_folder')

    # Add Faiss index to the "embeddings" column
    loaded_dataset.add_faiss_index(column="embeddings")
    question_embedding = get_embeddings([query]).numpy()

    # Use the Faiss index to find nearest neighbors
    scores, samples = loaded_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=5
    )

    # Create a DataFrame to display the results
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    Title=[]
    rows=[]
    prompt="Give a generic information on the subject under 80 words."
    # skip_len=len(prompt)
    prompt+=query
    llm_tokenize,llm_model,_1,_2=load_model()
    flag=0

    for _, row in samples_df.iterrows():
        # print(f"Series Title: {row.Title}")
        # print(f"Overview: {row.Abstract}")
        print(row.scores)
        if row.scores<50:
            flag=1
            Title.append(row.Title)
            rows.append(row.Abstract)
    if flag==1:
        generate_text(prompt,llm_model,llm_tokenize)
    if flag==0:
        st.warning("No relevant Information available")
    return Title,rows
def generate_text(input_text, model, tokenizer, max_length=250):
    # Encode the input text
    # dy_text=st.empty()
    skip_len=len(input_text)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate output text
    output_ids = model.generate(input_ids, max_length=max_length,temperature=0.7,do_sample=True)

    # Decode the output text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text=output_text[skip_len:]
    output_text=trim_to_full_sentence(output_text,300)
    st.text_area("General Knowledge related to your search:",output_text+' .',height=250)

    st.markdown("----")
    
def trim_to_full_sentence(text, word_limit):
    words = text.split()
    
    # If the text is shorter than the limit, return it as is
    if len(words) <= word_limit:
        return text

    trimmed_text = ' '.join(words[:word_limit])
    
    # Find the last period that ends a sentence
    last_period = trimmed_text.rfind('.')
    
    # If there's no period, return the text as is
    if last_period == -1:
        return trimmed_text

    # Return the text up to the last complete sentence
    return trimmed_text[:last_period + 1]
    
@st.cache_resource(show_spinner=False)
def load_model(spinner=False):
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer_semantics = AutoTokenizer.from_pretrained(model_ckpt)
    model_semantics = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
    
    # Path to your fine-tuned model
    model_path ="gpt2-finetuned"

    # Load the tokenizer and model
    tokenizer_llm = GPT2Tokenizer.from_pretrained(model_path)
    model_llm = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer_llm,model_llm,tokenizer_semantics,model_semantics
