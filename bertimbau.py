# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:48:02 2025

@author: mlfarias
"""

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import os

# Caminho para o diretório local onde você salvou o BERTimbau
model_path = './portuguese-bert-master/ner_evaluation/'  # ajuste para o caminho onde você salvou

# Verifica se os arquivos necessários estão no diretório
assert os.path.exists(os.path.join(model_path, 'pytorch_model.bin')), "Modelo não encontrado"
assert os.path.exists(os.path.join(model_path, 'config.json')), "Configuração não encontrada"
assert os.path.exists(os.path.join(model_path, 'vocab.txt')), "Vocabulário não encontrado"

# Carrega o tokenizer e o modelo do diretório local
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# Função para transformar texto em vetor (embedding)
def embed(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Sua pergunta
pergunta = "Qual é o prazo para entrega da declaração do IRPF?"

# Base de respostas possíveis
respostas = [
    "A declaração do Imposto de Renda da Pessoa Física deve ser entregue até 30 de abril.",
    "O ICMS é um imposto estadual cobrado sobre circulação de mercadorias.",
    "O Simples Nacional é um regime de tributação para pequenas empresas.",
    "A nota fiscal eletrônica deve ser emitida por empresas do regime normal de tributação."
]

# Embed da pergunta
vec_pergunta = embed(pergunta, tokenizer, model)

# Busca da resposta mais similar
resposta_mais_proxima = None
maior_similaridade = -1

for resposta in respostas:
    vec_resposta = embed(resposta, tokenizer, model)
    sim = np.dot(vec_pergunta, vec_resposta) / (np.linalg.norm(vec_pergunta) * np.linalg.norm(vec_resposta))
    if sim > maior_similaridade:
        maior_similaridade = sim
        resposta_mais_proxima = resposta

# Resultado
print("\n🔎 Pergunta:", pergunta)
print("✅ Resposta mais próxima:", resposta_mais_proxima)
