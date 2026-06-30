T2 — Matching de Produtos (Aprendizado de Máquina)
Prof. Me. Otávio Parraga
Nomes e Matrículas:
Bernardo Klein - 22103012 
Bernardo Fiorese - 23102185 
Bruno Almeida - 22180462 
Bruno Roese - 24280180
João Aiolfi - 22107503      
Rafael Prestes - 22280060

====================================================

Sistema de matching de produtos: dado um texto de consulta heterogêneo
(ex.: "FANTA LARANJA 2L C/6"), encontra o produto correspondente no
catálogo normalizado (Dados/catalog.csv, 14.206 produtos).

Duas abordagens principais — mais uma variante com LLM:
  Abordagem 1 (NLP clássico) — TF-IDF (palavras e n-gramas de caracteres)
      com similaridade de cosseno e BM25 (implementação própria).
  Abordagem 2 (Deep Learning) — embeddings semânticos com o modelo
      sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, nas
      variantes busca densa pura e híbrida (BM25 top-50 -> re-rank neural).
  Abordagem 3 - Inclui ainda uma variante com LLM: re-rank por prompting zero-shot
      e few-shot (BM25 top-10 -> Google Gemini escolhe o produto).

-----------------------------------------------------------

1. Python 3.10+ (testado com 3.12).
2. Instalar as dependências:

       pip install -r requirements.txt

3. (Somente para a variante LLM da Abordagem 2) configurar a API key do
   Google Gemini — obtenha uma chave gratuita em
   https://aistudio.google.com/apikey e exporte-a como variável de ambiente:

       export GEMINI_API_KEY="sua_chave"      # Linux/Mac
       set GEMINI_API_KEY=sua_chave           # Windows CMD

   Sem a chave (ou sem o pacote google-genai), apenas a variante LLM é
   pulada; todo o restante (Abordagens 1 e 2-embeddings) roda normalmente.

-----------------------------------------------------------

Opção A — Notebook (recomendado para a apresentação):
    abrir T2_Matching_Produtos.ipynb (Jupyter/VS Code) a partir da raiz
    do projeto e executar todas as células (Run All). O notebook é
    autocontido: exploração, pré-processamento, as duas abordagens,
    avaliação val/teste, análise qualitativa, NO_MATCH e preenchimento
    de queries.csv (~3 min com o cache de embeddings já criado).

Opção B — Scripts, a partir da raiz do projeto (a pasta com src/ e Dados/):

1. Exploração dos dados:
       python src/explore.py

2. Abordagem 1 (clássica) — métricas na validação e no teste:
       python src/approach1_classic.py --split val
       python src/approach1_classic.py --split test

3. Abordagem 2 (deep learning) — métricas na validação e no teste:
       python src/approach2_deep.py --split val
       python src/approach2_deep.py --split test
   (a primeira execução codifica o catálogo e grava cache em cache/;
    as execuções seguintes reutilizam o cache)

   Ablação sem pré-processamento (apêndice do relatório):
       python src/approach2_deep.py --split val --raw-text

   Variante LLM (zero/few-shot, requer GEMINI_API_KEY) — avaliada em
   amostra por causa do limite de requisições do free tier:
       python src/approach2_llm.py --split val --shot all --sample 15

4. Análise qualitativa (gera results/analise_qualitativa_test.md):
       python src/qualitative.py --split test --classic tfidf_char --deep deep_hybrid

5. Comportamento em casos NO_MATCH (gera results/no_match.md):
       python src/run_no_match.py

6. Preenchimento de queries.csv — "matched_id a ser preenchido pelo
   grupo" (gera results/queries_preenchido.csv com text, matched_id e
   score do top-1; o arquivo original em Dados/ não é modificado):
       python src/fill_queries.py

ESTRUTURA DO CÓDIGO
-----------------------------------------------------------
T2_Matching_Produtos.ipynb  notebook autocontido com todo o pipeline
src/preprocess.py         pré-processamento textual (normalização)
src/bm25.py               implementação própria do BM25 (Okapi)
src/evaluate.py           métricas P@1, MRR@5 e R@5
src/approach1_classic.py  Abordagem 1: TF-IDF (cosseno) e BM25
src/approach2_deep.py     Abordagem 2: embeddings (dense e híbrido)
src/approach2_llm.py      Abordagem 2 (variante LLM): re-rank zero/few-shot (Gemini)
src/explore.py            exploração dos dados
src/qualitative.py        análise qualitativa (acertos/erros/ambíguos)
src/run_no_match.py       probe de casos NO_MATCH
src/fill_queries.py       preenchimento em lote de queries.csv
results/                  saídas (top-5 por query, análises em Markdown)

MÉTRICAS FINAIS (queries_test.csv, 250 queries)
--------------------------------------------------------------
Abordagem 1 — TF-IDF char (3-5):  P@1=0.992  MRR@5=0.996  R@5=1.000
Abordagem 1 — BM25:               P@1=0.988  MRR@5=0.992  R@5=1.000
Abordagem 2 — híbrida (BM25+NN):  P@1=0.944  MRR@5=0.962  R@5=0.984
(empates de score são resolvidos deterministicamente pela ordem do
catálogo — ordenação estável; ver seção 3.3 do relatório)

A variante LLM (zero/few-shot) é avaliada em amostra da validação (depende
de API e do limite do free tier), não nas 250 do teste — ver a Seção 4 do
relatório.
