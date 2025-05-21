# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:04:42 2025

@author: mlfarias e yfdantas

Fontes úteis:
    https://huggingface.co/pierreguillou/bert-base-cased-squad-v1.1-portuguese
    
    https://medium.com/@pierre_guillou/nlp-modelo-de-question-answering-em-qualquer-idioma-baseado-no-bert-base-estudo-de-caso-em-12093d385e78
    
    https://github.com/piegu
"""

import os
import re
import pdfplumber
from nltk import tokenize
from fuzzywuzzy import fuzz
from string import punctuation
from nltk.corpus import stopwords
from transformers import pipeline

import warnings
warnings.filterwarnings('ignore')

chunk_size = 512

model_name = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
nlp = pipeline('question-answering', model = model_name)

stopwords = set(stopwords.words('portuguese') + list(punctuation))



#Extração dos PDFs 
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
pdf_files = collect_pdf_files(['..\Legislação', '..\Manuais'])

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

def buscar_trecho(perg, base_trechos):
    pergunta = perg
    pergunta = pergunta.replace('?', '')
    pergunta = pergunta.replace('qual', '')
    pergunta = pergunta.replace('quem', '')
    pergunta = pergunta.replace('que', '')
    pergunta = pergunta.replace('onde', '')
    pergunta = pergunta.replace('sergipe', '')
    token = tokenize.word_tokenize(pergunta, language='portuguese')
    pergunta = [palavra for palavra in token if palavra not in stopwords]
    
    trecho = ""

    for entrada in base_trechos:
        for termo in pergunta:
            for palavra in entrada:
                x = fuzz.ratio(palavra,termo)
                if x > 0.8:
                    trecho = trecho + " " + entrada
                    break
    
    return trecho

final_text = ""
# Processando os PDFs e aplicando NER
for pdf in pdf_files:
    print(f"\nProcessando: {pdf}\n{'-'*50}")
    text = extract_text_from_pdf(pdf)
    preprocessed_text = preprocess_text(text)
    final_text += preprocessed_text.lower()

base_text = [final_text[i:i+chunk_size] for i in range(0, len(final_text), chunk_size)]


print("#######################################################################################")
while True:
    pergunta = input("\n> Sua pergunta: ").strip()
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("👋 Encerrando.")
        break
    
    
    trecho = buscar_trecho(pergunta.lower(), base_text)
    print('> Varrimento de texto concluído!')
    
    resposta = nlp(pergunta, trecho)
    
    if resposta['score'] > 0.1:
        print("Resposta: %s"%resposta['answer'])
    else: 
        print('Resposta: Não foi possível recuperar essa informação com segurança em nossa base de dados!')



