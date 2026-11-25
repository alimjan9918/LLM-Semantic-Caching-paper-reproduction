"""Download pinned RouterBench 0-shot data and a small CPU embedding model."""
import argparse
import hashlib
import json
from pathlib import Path

from real_world import DATA_ROOT, DATASET_REVISION, MODEL_REVISION


def download(root=DATA_ROOT):
    from huggingface_hub import hf_hub_download
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [('withmartian/routerbench', 'dataset', DATASET_REVISION,
             'routerbench_0shot.pkl', root),
            ('withmartian/routerbench', 'dataset', DATASET_REVISION, 'README.md', root / 'source-card')]
    jobs += [('Xenova/all-MiniLM-L6-v2', 'model', MODEL_REVISION, filename, root / 'embedding-model')
             for filename in ('onnx/model_quantized.onnx', 'tokenizer.json',
                              'tokenizer_config.json', 'config.json', 'README.md')]
    manifest = {'dataset': 'withmartian/routerbench', 'dataset_revision': DATASET_REVISION,
                'embedding_model': 'Xenova/all-MiniLM-L6-v2', 'embedding_revision': MODEL_REVISION,
                'files': []}
    for repo, kind, revision, filename, folder in jobs:
        print(f'Downloading {repo}/{filename}', flush=True)
        path = Path(hf_hub_download(repo_id=repo, repo_type=kind, revision=revision,
                                    filename=filename, local_dir=folder))
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        manifest['files'].append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                                      sha256=digest, repository=repo, revision=revision, filename=filename))
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print(f'{path.name}: {path.stat().st_size:,} bytes, sha256 {digest}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT)
    download(parser.parse_args().data_dir)
