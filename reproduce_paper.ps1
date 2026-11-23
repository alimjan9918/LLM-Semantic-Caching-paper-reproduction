param(
    [int]$Runs = 10,
    [string]$Out = "results\paper_reproduction"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Model = Join-Path $Root "models\paraphrase-MiniLM-L6-v2"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Model)) {
    throw "Embedding model not found at $Model. See REPRODUCE.md for download instructions."
}

$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
$TikTokenCache = Join-Path $Root "tiktoken_cache"

$env:SEMANTIC_CACHE_EMBEDDING_MODEL = $Model
$env:SEMANTIC_CACHE_LOCAL_FILES_ONLY = "1"
$env:SEMANTIC_CACHE_TOKENIZER = "tiktoken"
$env:TIKTOKEN_CACHE_DIR = $TikTokenCache
$env:TOKENIZERS_PARALLELISM = "false"

& $Python (Join-Path $Root "run.py") --runs $Runs --out (Join-Path $Root $Out)
