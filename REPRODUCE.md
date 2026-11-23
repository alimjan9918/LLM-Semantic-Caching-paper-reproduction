# Reproduce the six-panel paper figure

The reproduction uses the paper's first 20 queries, all 10 seeds, and the
following settings:

- embedding model: `paraphrase-MiniLM-L6-v2`
- tokenizer: GPT-2 via `tiktoken`
- `m = 20`, `k = 5`, `T = 200`, `delta = 0.8`
- offline stream lengths: 300 to 600

## 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Obtain the embedding model

The model is available from the Sentence-Transformers authors:

```powershell
New-Item -ItemType Directory models -Force
curl.exe -L --fail -o models\paraphrase-MiniLM-L6-v2.zip `
  https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/paraphrase-MiniLM-L6-v2.zip
Expand-Archive models\paraphrase-MiniLM-L6-v2.zip `
  -DestinationPath models\paraphrase-MiniLM-L6-v2
```

## 3. Run all experiments

```powershell
$env:SEMANTIC_CACHE_EMBEDDING_MODEL = "$PWD\models\paraphrase-MiniLM-L6-v2"
$env:SEMANTIC_CACHE_LOCAL_FILES_ONLY = "1"
$env:SEMANTIC_CACHE_TOKENIZER = "tiktoken"
.\.venv\Scripts\python.exe run.py --runs 10 --out results\paper_reproduction
```

Or use the supplied one-command wrapper:

```powershell
.\reproduce_paper.ps1
```

The run writes:

- `results/paper_reproduction/figure.pdf`
- `results/paper_reproduction/figure.png`
- one JSON file for each experiment

## Notes

- The loss, regret, and switch-count results reproduce the paper.
- Runtime values depend on the machine, Python version, and CPU load. The
  relative ordering and the very low runtime of `CLCB-SC-LS` are preserved.
- `plot.py` now saves both PDF and PNG, so the paper figure can be used
  directly in a manuscript or report.
