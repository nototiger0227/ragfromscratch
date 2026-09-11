# FinRAG — Stage 1: minimal RAG pipeline

A bare-bones but complete RAG pipeline for querying financial PDFs.
Five files, each doing one clear job:

| File | Job |
|---|---|
| `config.py` | Settings in one place |
| `extract.py` | PDF → text |
| `chunk.py` | text → list of chunks |
| `ingest.py` | chunks → embeddings → Chroma (vector DB) |
| `query.py` | question → retrieve chunks → ask LLM → answer |

## Setup

```bash
cd finrag-stage1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
# then open .env and paste in a free Gemini API key from https://aistudio.google.com/apikey
```

## Run it

```bash
# 1. Put a PDF (e.g. a 10-K) in data/, then index it:
python ingest.py data/apple_2024.pdf apple_2024

# 2. Ask questions about it:
python query.py
> What was total revenue?
> What risks does the company mention around supply chain?
```

## What to actually read, in order

1. **`extract.py`** — trivial, just confirms "PDF → string" is the whole first step.
2. **`chunk.py`** — run it standalone (`python chunk.py`) to *see* the overlap
   happen on a toy string. This is the single easiest-to-misunderstand concept
   in RAG; watching it on 150 characters before trusting it on a real PDF helps.
3. **`ingest.py`** — read `embed_texts` and think about what a vector *is*
   (just a list of floats) before reading how Chroma stores it.
4. **`query.py`** — read `retrieve()` then `answer()`. Notice retrieval and
   generation are two totally separate steps — you can test/debug them
   independently. Try calling `retrieve("your question")` alone in a Python
   shell and print the chunks it returns *before* trusting what the LLM says
   about them. This is the #1 debugging move in RAG: when the answer is bad,
   check retrieval first — 90% of RAG bugs are retrieval bugs, not the LLM's fault.

## Exercises before moving to Stage 2

Try these — they'll surface the real weaknesses this naive version has:

- Ask a question whose answer is a specific number (e.g. "what was net
  income in millions"). Does chunking ever cut the number away from its
  label? Print the raw retrieved chunks to check.
- Ask a question using words that don't appear in the document at all
  (e.g. ask about "profitability" when the doc only says "net income").
  Does semantic search still find it? This is exactly what vector search
  is good at that keyword search isn't.
- Now ask a question with an exact ticker symbol or unusual proper noun.
  Does it retrieve well? (Often *worse* — this is exactly the gap that
  BM25/keyword search fills, which is Stage 2.)
- Try `CHUNK_SIZE = 300` vs `CHUNK_SIZE = 2000` in `config.py`, re-ingest,
  and compare answer quality. This is the central RAG tuning tradeoff:
  small chunks = precise but low-context; large chunks = rich context but
  mushy vectors.

## Stage 2 preview (once you've done the exercises)

- Replace character-count chunking with structure-aware chunking
  (split on headings/paragraphs first).
- Add BM25 (keyword) retrieval alongside the vector search, and merge
  the two result sets ("hybrid retrieval").
- Add a cross-encoder reranker to re-score the merged candidates before
  picking the final top-k — this is what actually fixes most of the bad
  retrievals you'll have found in the exercises above.

Ask me when you're ready and we'll build that layer on top of this,
file by file, the same way.
