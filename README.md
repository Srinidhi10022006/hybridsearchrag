# Hybrid Search RAG: BM25 + Dense Retrieval on MS MARCO

An end-to-end hybrid retrieval system that combines lexical search (BM25) with dense semantic search (FAISS), evaluated on MS MARCO. Runs fully locally: no API keys, no paid services.

## Key findings

- Dense retrieval beat BM25 on every metric.
- A naive 50/50 hybrid fusion **underperformed pure dense retrieval**.
- Tuning the fusion weight (alpha = 0.2 to 0.3) made the hybrid beat both methods, with the best Recall@5 and about **37% higher MRR than BM25 alone**.

Takeaway: hybrid search is not automatically better. The fusion weight has to be tuned against an evaluation set.

## How it works

1. Load an MS MARCO subset from Hugging Face.
2. Build a BM25 index (lexical scores) and a dense embedding index with FAISS (semantic scores).
3. Combine the two with a weighted score: `score = alpha * dense + (1 - alpha) * bm25`, after normalizing each.
4. Sweep alpha and compare BM25, Dense and Hybrid on the same queries.

**Metrics:** Precision@5, Recall@5, MRR, retrieval time.

## Results

Full numbers are in the CSVs:

- `comparison_results.csv`: BM25 vs Dense vs Hybrid
- `alpha_sweep_results.csv`: metrics across alpha values

<!-- Optional: paste a small table of the best rows here so recruiters see numbers without opening the CSVs. -->

## Repo layout

| File | Purpose |
|---|---|
| `jupyter.ipynb` | Full pipeline: data loading, indexing, fusion, evaluation, alpha sweep |
| `app.py` | Streamlit app with two tabs: live search comparison and evaluation dashboard |
| `comparison_results.csv` | Method comparison results |
| `alpha_sweep_results.csv` | Alpha sweep results |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook              # run jupyter.ipynb top to bottom
streamlit run app.py
```

The notebook generates `faiss_index.bin` and `saved_data.pkl` (the index and processed data). They are not committed, so run the notebook once before launching the app.

## Stack

Python, Hugging Face Datasets, BM25, FAISS, sentence embeddings, Streamlit, pandas.
