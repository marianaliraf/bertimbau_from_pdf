# -*- coding: utf-8 -*-
"""
Created on Thu Jan  15 15:15:20 2025

@authors: mlfarias & yurifarod
"""

import os
import re
import pdfplumber
from transformers import BertTokenizer, BertForTokenClassification, pipeline


# Coletando PDFs das pastas especificadas
def collect_pdf_files(directories):
    pdf_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    return pdf_files

# Coletando arquivos das pastas 'Legislação' e 'Manuais'
pdf_files = collect_pdf_files(['../Legislação', '../Manuais'])

# extrair texto dos documentos PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

#pré-processamento do texto
def preprocess_text(text):
    text = text.replace('\r', ' ').replace('\n', ' ') 
    text = re.sub(r'[_\-\=\*]{2,}', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text


# Função para dividir o texto respeitando o limite de 512 tokens
def split_text(text, tokenizer, max_length=256):
    tokens = tokenizer(
        text,
        truncation=True, 
        padding="max_length",  
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"][0]
    for i in range(0, len(input_ids), max_length):
        chunk_ids = input_ids[i:i + max_length]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        yield chunk_text

# Carregando o modelo BERTimbau específico para NER
#tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
#model = BertForTokenClassification.from_pretrained("neuralmind/bert-large-portuguese-cased")

model = BertForTokenClassification.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

# Pipeline para NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Processando os PDFs e aplicando NER
for pdf in pdf_files:
    print(f"\nProcessando: {pdf}\n{'-'*50}")
    text = extract_text_from_pdf(pdf)
    preprocessed_text = preprocess_text(text)
    
    for chunk in split_text(preprocessed_text, tokenizer):
        entities = ner_pipeline(chunk)
        
        for entity in entities:
            print(f"Entidade: {entity['word']}, Tipo: {entity['entity_group']}, Confiança: {entity['score']:.2f}")
