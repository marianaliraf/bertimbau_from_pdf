# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:04:42 2025

@author: mlfarias e @yfdantas
"""

import os
import fitz  
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


PASTAS_PDF = ['Legislação', 'Manuais']
MODEL_PATH = './portuguese-bert-master/ner_evaluation/'


#Extração dos PDFs 
def extrair_texto(caminho_pdf):
    try:
        doc = fitz.open(caminho_pdf)
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
        return texto
    except Exception as e:
        print(f"[ERRO] Falha ao ler {caminho_pdf}: {e}")
        return ""

def ler_pdfs(pastas):
    textos_por_arquivo = {}
    for pasta in pastas:
        for raiz, _, arquivos in os.walk(pasta):
            for nome_arquivo in arquivos:
                if nome_arquivo.endswith(".pdf"):
                    caminho_pdf = os.path.join(raiz, nome_arquivo)
                    print(f"{caminho_pdf}")
                    texto = extrair_texto(caminho_pdf)
                    textos_por_arquivo[nome_arquivo] = texto
    return textos_por_arquivo



def dividir_stucks(texto, tamanho_maximo=250):
    paragrafos = texto.split('\n')
    trechos = []
    bloco = ""

    for par in paragrafos:
        if len(par.strip()) == 0:
            continue
        if len(bloco) + len(par) < tamanho_maximo:
            bloco += par.strip() + " "
        else:
            trechos.append(bloco.strip())
            bloco = par.strip() + " "
    if bloco:
        trechos.append(bloco.strip())

    return trechos

def gerar_embedding(texto):
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()



tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)


textos_extraidos = ler_pdfs(PASTAS_PDF)

todos_os_trechos = []

for nome_arquivo, texto in textos_extraidos.items():
    trechos = dividir_stucks(texto)
    for trecho in trechos:
        embedding = gerar_embedding(trecho)
        todos_os_trechos.append({
            "arquivo": nome_arquivo,
            "trecho": trecho,
            "embedding": embedding
        })



def buscar_resposta(pergunta, base_trechos):
    emb_pergunta = gerar_embedding(pergunta)
    melhor = None
    maior_sim = -1

    for entrada in base_trechos:
        emb_trecho = entrada["embedding"]
        sim = np.dot(emb_pergunta, emb_trecho) / (np.linalg.norm(emb_pergunta) * np.linalg.norm(emb_trecho))
        if sim > maior_sim:
            maior_sim = sim
            melhor = entrada

    return melhor, maior_sim

print("\nDigite sua pergunta (ou 'sair' para encerrar):")

while True:
    pergunta = input("\n> Sua pergunta: ").strip()
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("👋 Encerrando.")
        break

    resposta, similaridade = buscar_resposta(pergunta, todos_os_trechos)
    print("\n Trecho mais similar:")
    '''print(f"Arquivo: {resposta['arquivo']}")
    print(f"Similaridade: {similaridade:.4f}")'''
    print(f"\n Trecho:\n{resposta['trecho']}")


