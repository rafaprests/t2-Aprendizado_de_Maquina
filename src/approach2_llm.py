# -*- coding: utf-8 -*-
"""
Abordagem 2 — variante LLM: re-rank por prompting zero-shot / few-shot.

Estratégia recomendada no enunciado (cascata): o BM25 filtra os top-N candidatos
baratos e um LLM faz a DECISÃO FINAL, escolhendo o produto correto entre eles.
Enviar só ~10 nomes (e não os 14.206 do catálogo) corta o custo de tokens e foca
o modelo nos casos plausíveis.

  - zero-shot: o prompt traz só a query e os candidatos, sem exemplos resolvidos;
  - few-shot : o prompt inclui alguns pares (query -> produto correto) para o
               modelo calibrar o formato e o estilo ruidoso das queries.

O LLM devolve um ÚNICO vencedor; para manter as métricas de lista (MRR@5/R@5)
montamos o top-5 com a escolha do LLM em 1º e o restante na ordem do BM25.

Provedor: Google Gemini. Requer o pacote google-genai e a variável de ambiente
GEMINI_API_KEY. Configure a chave antes de rodar:
    export GEMINI_API_KEY="sua_chave"          # Linux/Mac
    set GEMINI_API_KEY=sua_chave               # Windows CMD

Observação sobre cota: o free tier limita as requisições por minuto (~5-15), o
que torna a avaliação lenta; por isso usamos uma AMOSTRA (--sample) por padrão.

Uso:
    python src/approach2_llm.py --split val                  # zero + few (amostra)
    python src/approach2_llm.py --split val --shot few --sample 30
    python src/approach2_llm.py --split test --model gemini-2.5-flash --sample 50
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from approach1_classic import load_catalog, make_bm25_ranker, save_details
from evaluate import evaluate_ranker, fmt

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
RESULTS = ROOT / "results"

# gemini-2.5-flash-lite: free tier com mais vazão que o 2.5-flash (5 req/min) e
# cota diária própria; suficiente para escolher entre ~10 candidatos.
DEFAULT_MODEL = "gemini-2.5-flash-lite"
N_CANDIDATES = 10            # top-N do BM25 enviados ao LLM (sugestão do enunciado)
N_FEW_SHOT = 4              # nº de exemplos resolvidos no prompt few-shot


def get_client():
    """Cria o cliente Gemini, ou None se faltar o pacote/variável de ambiente."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY não definida — configure a chave para rodar o LLM.")
        return None
    try:
        from google import genai
    except ImportError:
        print("pacote google-genai ausente — instale com: pip install google-genai")
        return None
    return genai.Client(api_key=key)


def build_few_shot(val, id2name, n=N_FEW_SHOT):
    """Exemplos (query -> produto correto) tirados das ÚLTIMAS linhas da validação.

    São fixos e poucos; ao avaliar val, essas linhas são excluídas da amostra
    (em main) para não vazar resposta.
    """
    idx = list(range(len(val) - n, len(val)))
    examples = [(val.iloc[i]["text"], id2name.get(val.iloc[i]["matched_id"], ""))
                for i in idx]
    return idx, examples


def build_prompt(query, cand_names, few_shot_examples):
    """Lista numerada de candidatos + pedido para o LLM responder só o número."""
    instr = (
        "Você faz matching de produtos de varejo. Dada uma CONSULTA (texto de pedido, "
        "abreviado e ruidoso) e uma lista numerada de PRODUTOS do catálogo, responda "
        "APENAS com o número do produto que corresponde à consulta. "
        "Se nenhum corresponder, responda 0.\n\n"
    )
    shots = ""
    if few_shot_examples:
        # cada exemplo mostra o formato esperado e uma resposta correta
        for q, name in few_shot_examples:
            shots += f"CONSULTA: {q}\nPRODUTOS:\n1. {name}\nResposta: 1\n\n"
    cands = "\n".join(f"{j + 1}. {n}" for j, n in enumerate(cand_names))
    return f"{instr}{shots}CONSULTA: {query}\nPRODUTOS:\n{cands}\nResposta:"


def make_llm_ranker(client, model, catalog, bm25_rank, few_shot_examples=None,
                    n_candidates=N_CANDIDATES, pace=0.0):
    """Re-ranker LLM: BM25 filtra n_candidates -> o LLM escolhe o 1º lugar.

    pace: segundos de espera antes de cada chamada (para respeitar o limite por
    minuto do free tier). O retry trata 429/5xx; respostas são memorizadas.
    """
    id2name = dict(zip(catalog["product_id"], catalog["product_name"]))
    cfg = {"temperature": 0.0, "max_output_tokens": 10}
    if "2.5" in model:
        cfg["thinking_config"] = {"thinking_budget": 0}   # família 2.5: desliga o "thinking"
    cache = {}

    def choose(query, cand_names):
        key = (query, tuple(cand_names))
        if key in cache:
            return cache[key]
        if pace:
            time.sleep(pace)
        resp = None
        for _ in range(6):
            try:
                resp = client.models.generate_content(
                    model=model, contents=build_prompt(query, cand_names, few_shot_examples),
                    config=cfg)
                break
            except Exception as e:                       # transitórios: 429 e 5xx/timeout
                msg = str(e)
                if any(t in msg for t in ("RESOURCE_EXHAUSTED", "429", "500", "502",
                                          "503", "UNAVAILABLE", "timed out", "timeout")):
                    time.sleep(15)
                    continue
                raise
        m = re.search(r"\d+", resp.text or "") if resp is not None else None
        idx = int(m.group()) if m else 0
        cache[key] = idx
        return idx

    def rank(text, k=5):
        cand = bm25_rank(text, n_candidates)             # [(pid, score), ...]
        cand_ids = [pid for pid, _ in cand]
        names = [id2name.get(pid, "") for pid in cand_ids]
        choice = choose(text, names)                     # 1..N ou 0
        if 1 <= choice <= len(cand_ids):
            pick = cand_ids[choice - 1]
            # escolha do LLM em 1º; o resto mantém a ordem do BM25 (preserva R@5/MRR@5)
            ordered = [pick] + [c for c in cand_ids if c != pick]
        else:
            ordered = cand_ids                           # LLM disse "nenhum" -> ordem BM25
        # score sintético decrescente só para manter o formato (pid, score)
        return [(pid, 1.0 - j * 0.01) for j, pid in enumerate(ordered[:k])]

    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--shot", choices=["zero", "few", "all"], default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=15,
                    help="nº de queries avaliadas (free tier é lento; mantenha pequeno)")
    ap.add_argument("--candidates", type=int, default=N_CANDIDATES,
                    help="top-N do BM25 enviados ao LLM")
    ap.add_argument("--pace", type=float, default=4.5,
                    help="espera (s) entre chamadas, p/ não estourar o limite por minuto")
    args = ap.parse_args()

    client = get_client()
    if client is None:
        sys.exit(1)

    catalog = load_catalog(use_brand=True)
    id2name = dict(zip(catalog["product_id"], catalog["product_name"]))
    bm25_rank = make_bm25_ranker(catalog)

    val = pd.read_csv(DADOS / "queries_val.csv", dtype=str)
    fs_idx, fs_examples = build_few_shot(val, id2name)

    queries = pd.read_csv(DADOS / f"queries_{args.split}.csv", dtype=str)
    if args.split == "val":
        queries = queries.drop(index=fs_idx)             # evita vazamento dos exemplos
    sample = queries.head(args.sample)

    print(f"[abordagem 2 - LLM] split={args.split}  modelo={args.model}  "
          f"amostra={len(sample)}  candidatos={args.candidates}\n")

    shots = ["zero", "few"] if args.shot == "all" else [args.shot]
    for s in shots:
        examples = fs_examples if s == "few" else None
        ranker = make_llm_ranker(client, args.model, catalog, bm25_rank,
                                 few_shot_examples=examples,
                                 n_candidates=args.candidates, pace=args.pace)
        metrics, details, _ = evaluate_ranker(ranker, sample)
        print(f"llm_{s:<5} {fmt(metrics)}")
        save_details(details, catalog,
                     RESULTS / f"detalhes_llm_{s}_{args.split}.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
