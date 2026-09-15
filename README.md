# AI Customer Support Agent for Tesco Twitter Support

This is my Hiver SDE Intern take-home submission. It builds a support agent for **Tesco** using the Kaggle `thoughtvector/customer-support-on-twitter` dataset.

The agent does three things for an incoming customer tweet:

1. classifies the message into a small Tesco-specific intent taxonomy,
2. drafts a reply grounded in retrieved historical Tesco support resolutions,
3. decides whether to auto-handle or escalate, with a reason.

The important part is the proof. This repo includes a 200-example hand-labelled golden set, automated evaluation, two baselines, LLM-as-judge reply scoring, and judge-vs-human agreement.

## Headline Results

Computed on `data/golden/golden_set.csv` with `python -m src.evaluation.evaluate`.

| Metric | Trivial baseline | TF-IDF baseline | Final agent |
|---|---:|---:|---:|
| Intent accuracy | 0.170 | 0.480 | **0.785** |
| Intent macro F1 | 0.029 | 0.439 | **0.789** |
| Escalation precision | 0.605 | - | 0.738 |
| Escalation recall | 1.000 | - | 0.909 |
| Escalation F1 | 0.754 | - | **0.815** |
| Missed escalation rate | 0.000 | - | 0.091 |
| False escalation rate | 1.000 | - | 0.494 |

Reply-quality judge results are in `results/judge_results.csv`. The LLM judge successfully scored 115 examples with mean scores: correctness 4.27, relevance 4.53, grounding 4.09, completeness 3.47, helpfulness 3.82.

Judge/human agreement over those 115 examples is in `results/judge_human_agreement.json`; Spearman correlations range from 0.913 to 0.982, and Cohen's kappa ranges from 0.748 to 0.952.

## Quick Reproduction Under 15 Minutes

These commands reproduce the headline metrics from the included compact artifacts and cached agent outputs. They do not require the full Kaggle dataset or an LLM API key.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.evaluation.evaluate
python -m src.evaluation.judge_agreement
python -m pytest
```

On macOS/Linux, use `source .venv/bin/activate` instead of the Windows activation command.

## Try The Agent

```bash
python -m src.agent --message "My food delivery is late again and nobody has called me back"
```

If the sentence-transformer embedding model is not already cached locally, retrieval falls back to a local TF-IDF retriever over `data/processed/retrieval_corpus.json`. If no LLM provider is configured, generation falls back to a deterministic privacy-safe reply instead of copying historical responses.

To use an LLM, create a local `.env` file:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
```

Supported providers in the code are `groq`, `openai`, `gemini`, and `none`.

## Repository Map

`src/agent.py` is the end-to-end entrypoint. `src/intents/` discovers and applies the intent taxonomy. `src/retrieval/` builds and serves historical-case retrieval. `src/generation/` drafts grounded replies and handles LLM fallbacks. `src/escalation/` contains the auto-handle/escalation policy. `src/evaluation/` contains the metrics, leakage checks, judge, judge-human agreement, and failure analysis.

`data/golden/golden_set.csv` is the 200-example hand-labelled golden set. `data/golden/annotation_guidelines.md` explains the sampling and labelling rules. `REPORT.md` is the six-page-style report, and `DECISIONS.md` is the decision log.

`data/processed/` includes compact artifacts needed for fast reproduction. The full Kaggle dump and the 800MB cleaned export are intentionally ignored.

## Full Pipeline

If you have the full Kaggle dataset at `data/raw/twcs/twcs.csv`, run:

```bash
python -m scripts.setup_data
python -m scripts.run_pipeline
python -m src.evaluation.evaluate --fresh
python -m src.evaluation.judge
python -m src.evaluation.judge_agreement
python -m src.evaluation.failure_analysis
```

The full fresh evaluation can take longer if an LLM is enabled, so the default reproduction path uses cached outputs.