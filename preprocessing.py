"""Extract one model's RouterBench records into the plain CSV consumed by e2.

python preprocessing.py --list-models
python preprocessing.py --model gpt-3.5-turbo-1106
python preprocessing.py --model gpt-3.5-turbo-1106 --queries 100 --seed 0

No embedding or generative model is needed for this extraction step.
"""
import argparse
import hashlib
import json
from pathlib import Path

from real_world import DATA_ROOT, DEFAULT_CSV, DEFAULT_MODEL, available_models, select_records


def preprocess(source=DATA_ROOT / 'routerbench_0shot.pkl', output=DEFAULT_CSV, *,
               model=DEFAULT_MODEL, n_queries=None, seed=0, eval_name=None):
    import pandas as pd
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError(f'raw dataset not found: {source}; see README.md for data setup')
    if source.resolve() == output.resolve() or output.suffix.lower() != '.csv':
        raise ValueError('output must be a new CSV path distinct from the raw source')
    records = select_records(pd.read_pickle(source), model, n_queries=n_queries,
                              seed=seed, eval_name=eval_name)
    rows = pd.DataFrame({
        'query_id': range(len(records.prompts)), 'sample_id': records.sample_ids,
        'eval_name': records.eval_names, 'model': records.model,
        'prompt': records.prompts, 'answer': records.answers, 'cost_usd': records.cost_usd,
        'query_bytes': records.query_bytes.astype(int), 'answer_bytes': records.answer_bytes.astype(int),
        'storage_bytes': (records.query_bytes + records.answer_bytes).astype(int),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding='utf-8-sig')
    with source.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = dict(source=str(source.resolve()), source_sha256=digest,
                    source_repository='https://huggingface.co/datasets/withmartian/routerbench',
                    model=records.model, n_queries=len(rows), seed=seed, eval_name=eval_name,
                    sampling='all valid distinct prompts in source order' if n_queries is None else 'seeded sampling without replacement',
                    storage_unit='UTF-8 bytes', cost_unit='recorded historical estimated USD',
                    cleaning='drop missing/empty text, nonfinite/nonpositive costs; deduplicate prompt, keep first',
                    columns=list(rows.columns))
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DATA_ROOT / 'routerbench_0shot.pkl')
    parser.add_argument('--output', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--queries', type=int, help='omit to retain all usable records for this model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-name', help='optional exact eval_name filter')
    parser.add_argument('--list-models', action='store_true', help='inspect source model names without writing CSV')
    args = parser.parse_args()
    try:
        if args.list_models:
            if not args.input.is_file():
                raise ValueError(f'raw dataset not found: {args.input}')
            frame = pd.read_pickle(args.input)
            print('\n'.join(available_models(frame)))
            return
        result = preprocess(args.input, args.output, model=args.model,
                             n_queries=args.queries, seed=args.seed, eval_name=args.eval_name)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Saved {result['n_queries']} queries for {result['model']}")
    print(args.output.resolve())
    print(args.output.with_suffix('.metadata.json').resolve())


if __name__ == '__main__':
    main()
