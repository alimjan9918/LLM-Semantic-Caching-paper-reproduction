"""RouterBench records and cached CPU sentence embeddings for e2.

No generative model or paid API is called. Recorded prices are historical
estimated USD costs, not measured inference latency.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from data import ReferenceProblem, arrival_probabilities
from storage import OnlineStorageModel

DATA_ROOT = Path(__file__).resolve().parent / 'real-world-dataset'
DATASET_REVISION = '784021482c3f320c6619ed4b3bb3b41a21424fcb'
MODEL_REVISION = '751bff37182d3f1213fa05d7196b954e230abad9'
DEFAULT_MODEL = 'gpt-3.5-turbo-1106'
DEFAULT_CSV = DATA_ROOT / 'routerbench_single_model.csv'


@dataclass
class RecordedQueries:
    prompts: list[str]
    answers: list[str]
    sample_ids: list[str]
    eval_names: list[str]
    cost_usd: np.ndarray
    query_bytes: np.ndarray
    answer_bytes: np.ndarray
    model: str


def available_models(frame):
    return sorted(col.removesuffix('|model_response') for col in frame.columns
                  if col.endswith('|model_response')
                  and col.removesuffix('|model_response') + '|total_cost' in frame.columns)


def select_records(frame, model, *, n_queries=20, seed=0, eval_name=None):
    """Filter before reproducible sampling; one distinct prompt per entry."""
    import pandas as pd
    if (n_queries is not None and n_queries < 2) or seed < 0:
        raise ValueError('require n_queries >= 2 and non-negative dataset seed')
    if model not in available_models(frame):
        raise ValueError(f'unknown model {model!r}; available models: {available_models(frame)}')
    response, cost = model + '|model_response', model + '|total_cost'
    columns = ['prompt', response, cost, 'sample_id', 'eval_name']
    if not set(columns) <= set(frame.columns):
        raise ValueError('RouterBench records lack prompt, sample_id or eval_name columns')
    rows = frame[columns].copy()
    if eval_name is not None:
        rows = rows[rows.eval_name == eval_name]
    rows[cost] = pd.to_numeric(rows[cost], errors='coerce')
    usable = (rows.prompt.map(lambda x: isinstance(x, str) and bool(x.strip()))
              & rows[response].map(lambda x: isinstance(x, str) and bool(x.strip()))
              & np.isfinite(rows[cost]) & (rows[cost] > 0))
    rows = rows[usable].drop_duplicates('prompt', keep='first')
    if len(rows) < 2 or (n_queries is not None and len(rows) < n_queries):
        raise ValueError(f'only {len(rows)} usable distinct records; requested {n_queries}')
    if n_queries is not None:
        chosen = np.random.default_rng(seed).choice(len(rows), size=n_queries, replace=False)
        rows = rows.iloc[chosen]
    prompts, answers = rows.prompt.tolist(), rows[response].tolist()
    return RecordedQueries(prompts, answers, rows.sample_id.astype(str).tolist(),
                           rows.eval_name.astype(str).tolist(), rows[cost].to_numpy(dtype=float),
                           np.array([len(x.encode('utf-8')) for x in prompts], dtype=float),
                           np.array([len(x.encode('utf-8')) for x in answers], dtype=float), model)


def read_records(path=DEFAULT_CSV, *, model=None,
                 n_queries=20, seed=0, eval_name=None):
    import pandas as pd
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'dataset missing: {path}; download routerbench_0shot.pkl and run preprocessing.py first')
    if path.suffix.lower() == '.csv':
        # Preserve strings such as NA/null and IDs with leading zeros.
        frame = pd.read_csv(path, keep_default_na=False, dtype=str, encoding='utf-8-sig')
        required = {'prompt', 'answer', 'model', 'sample_id', 'eval_name', 'cost_usd',
                    'query_bytes', 'answer_bytes', 'storage_bytes'}
        if not required <= set(frame.columns):
            raise ValueError(f'preprocessed CSV lacks columns: {sorted(required - set(frame.columns))}')
        models = frame.model.unique().tolist()
        if len(models) != 1 or not models[0]:
            raise ValueError('preprocessed CSV must contain exactly one model')
        if model is not None and model != models[0]:
            raise ValueError(f'CSV contains model {models[0]!r}, not {model!r}')
        model = models[0]
        query_bytes = frame.prompt.map(lambda x: len(x.encode('utf-8')))
        answer_bytes = frame.answer.map(lambda x: len(x.encode('utf-8')))
        for column, expected in [('query_bytes', query_bytes), ('answer_bytes', answer_bytes),
                                 ('storage_bytes', query_bytes + answer_bytes)]:
            values = pd.to_numeric(frame[column], errors='coerce')
            if not np.array_equal(values.to_numpy(), expected.to_numpy()):
                raise ValueError(f'CSV {column} sizes do not match UTF-8 text sizes')
        frame = frame.rename(columns={'answer': model + '|model_response',
                                      'cost_usd': model + '|total_cost'})
    elif path.suffix.lower() == '.pkl':
        # Read only the trusted official source; e2 normally uses the plain CSV.
        frame = pd.read_pickle(path)
        model = DEFAULT_MODEL if model is None else model
    else:
        raise ValueError('dataset must be a preprocessed .csv or official .pkl file')
    return select_records(frame, model, n_queries=n_queries,
                          seed=seed, eval_name=eval_name)


def encode_queries(prompts, model_dir, *, batch_size=16):
    """Quantized MiniLM on CPU: masked mean pooling and L2 normalization."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    model_dir = Path(model_dir)
    model_path = model_dir / 'onnx/model_quantized.onnx'
    tokenizer_path = model_dir / 'tokenizer.json'
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise ValueError('embedding files missing; see README.md for manual download links')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    input_names = {item.name for item in session.get_inputs()}
    vectors = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer.encode_batch(prompts[start:start + batch_size])
        inputs = {'input_ids': np.array([x.ids for x in encoded], dtype=np.int64),
                  'attention_mask': np.array([x.attention_mask for x in encoded], dtype=np.int64),
                  'token_type_ids': np.array([x.type_ids for x in encoded], dtype=np.int64)}
        outputs = session.run(None, {key: value for key, value in inputs.items() if key in input_names})
        hidden = outputs[0]
        mask = inputs['attention_mask'][..., None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        vectors.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
        print(f'[embedding] {min(start + batch_size, len(prompts))}/{len(prompts)} queries', flush=True)
    return np.concatenate(vectors).astype(np.float32)


def cached_embeddings(prompts, root=DATA_ROOT, *, encoder=None):
    root = Path(root)
    model_dir = root / 'embedding-model'
    # Hash actual local files too: manual replacements must invalidate vectors.
    local_hashes = {}
    for filename in ('onnx/model_quantized.onnx', 'tokenizer.json'):
        path = model_dir / filename
        if path.is_file():
            with path.open('rb') as handle:
                local_hashes[filename] = hashlib.file_digest(handle, 'sha256').hexdigest()
    fingerprint = json.dumps([MODEL_REVISION, local_hashes, 'masked-mean-l2-max256-v1', prompts],
                             ensure_ascii=False).encode('utf-8')
    cache = root / 'embeddings' / (hashlib.sha256(fingerprint).hexdigest() + '.npy')
    if cache.is_file():
        vectors = np.load(cache, allow_pickle=False)
    else:
        vectors = (encode_queries if encoder is None else encoder)(prompts, model_dir)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors, allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(prompts) or not np.isfinite(vectors).all():
        raise ValueError('invalid cached embedding shape or values')
    return vectors


def make_problem(records, embeddings, *, arrival='uniform', budget_fraction=0.25, budget=None):
    """Return hidden truth, byte storage and the USD-per-normalized-unit scale."""
    costs = np.asarray(records.cost_usd, dtype=float)
    vectors = np.asarray(embeddings, dtype=float)
    if (costs.ndim != 1 or costs.size < 2 or not np.isfinite(costs).all() or np.any(costs <= 0)
            or vectors.ndim != 2 or len(vectors) != len(costs) or not np.isfinite(vectors).all()):
        raise ValueError('invalid recorded costs or embeddings')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('zero query embedding')
    vectors = vectors / norms
    # Euclidean distance on embeddings divided by 2 is a metric in [0,1].
    squared = np.clip(2 - 2 * (vectors @ vectors.T), 0, 4)
    dist = np.sqrt(squared) / 2
    np.fill_diagonal(dist, 0)
    scale = 2.0 * float(costs.max())  # headroom for mean-preserving cost noise
    problem = ReferenceProblem(arrival_probabilities(len(costs), arrival), costs / scale,
                               dist, np.asarray(records.query_bytes))
    total = float((records.query_bytes + records.answer_bytes).sum())
    if not np.isfinite(budget_fraction) or not 0 < budget_fraction <= 1:
        raise ValueError('budget_fraction must lie in (0,1]')
    budget = budget_fraction * total if budget is None else float(budget)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('budget must be finite and positive')
    storage = OnlineStorageModel(records.query_bytes, records.answer_bytes, budget,
                                 budget / total, memory_noise=0.)
    return problem, storage, scale
