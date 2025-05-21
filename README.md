# 🔍 LLM Tributário - Sistema de Perguntas e Respostas em Documentos Tributários

Este projeto é um modelo de **Perguntas e Respostas (Question Answering)** desenvolvido em Python, baseado em modelos de linguagem (LLM) utilizando BERT em português. O sistema é capaz de responder perguntas relacionadas a documentos tributários, como legislações e manuais, extraindo informações diretamente de arquivos PDF.

## 🚀 Funcionalidades

- 🗂️ Leitura automática de arquivos PDF de pastas específicas (`Legislação` e `Manuais`);
- 🔍 Pré-processamento textual dos documentos;
- 🧠 Busca inteligente de trechos relevantes para a pergunta;
- 🤖 Respostas baseadas no modelo pré-treinado **BERT para SQuAD v1.1 em português**, disponibilizado no Hugging Face;
- 💬 Interface de linha de comando para realizar perguntas em linguagem natural.

## 🛠️ Tecnologias Utilizadas

- Python
- [Hugging Face Transformers](https://huggingface.co/)
- Modelo pré-treinado: [`pierreguillou/bert-base-cased-squad-v1.1-portuguese`](https://huggingface.co/pierreguillou/bert-base-cased-squad-v1.1-portuguese)
- PDFPlumber (extração de textos de PDFs)
- NLTK (tokenização e stopwords)
- Regex (limpeza de texto)

## ⚙️ Pré-requisitos

- Python 3.8 ou superior
- Instalar as bibliotecas necessárias:

```bash
pip install pdfplumber nltk transformers
```
Além disso, será necessário baixar os recursos de stopwords do NLTK:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📁 Estrutura Esperada das Pastas

O projeto espera encontrar arquivos PDF organizados nas seguintes pastas:

- `/Legislação`
- `/Manuais`

## 📜 Exemplo de Uso

```bash
> Sua pergunta: Qual é a base de cálculo do ICMS?
> Varrimento de texto concluído!
Resposta: A base de cálculo do ICMS corresponde ao valor da operação...
```

## 🔗 Fontes e Referências

- [Modelo BERT em português para SQuAD v1.1](https://huggingface.co/pierreguillou/bert-base-cased-squad-v1.1-portuguese)
- [Estudo de caso - Medium](https://medium.com/@pierre_guillou/nlp-modelo-de-question-answering-em-qualquer-idioma-baseado-no-bert-base-estudo-de-caso-em-12093d385e78)
- [GitHub de Pierre Guillou](https://github.com/piegu)

