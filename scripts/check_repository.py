"""Check the prospective Git payload without datasets, weights, or a GPU."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    proc = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                          cwd=ROOT,check=True,capture_output=True)
    names = sorted(set(proc.stdout.decode().split('\0'))-{''})
    assert names, 'No publishable files found'
    files = [ROOT/n for n in names]
    for p in files:
        assert p.exists(), f'Missing file: {p}'
        rel = p.relative_to(ROOT).as_posix()
        assert not rel.startswith(('adapters/','results/','data/evaluation/','data/prepared/'))
        assert 'worklog' not in p.name.lower(), f'Worklog inside payload: {rel}'
        assert p.suffix not in {'.jsonl','.safetensors','.gguf','.joblib','.pyc'}
        assert p.stat().st_size < 5_000_000, f'Unexpected large file: {rel}'
        if p.suffix == '.py': compile(p.read_text(),str(p),'exec')
        if p.suffix == '.json': json.loads(p.read_text())
        if p.suffix in {'.md','.json','.yaml','.yml','.py','.txt'}:
            text = p.read_text()
            if p.name != Path(__file__).name:
                assert not re.search(r'/(?:Users|home)/[A-Za-z0-9_-]+/',text), f'Personal absolute path: {rel}'
            assert not re.search(r'\bhf_[A-Za-z0-9]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b',text), f'Possible token: {rel}'
        if p.suffix == '.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if '://' in target or target.startswith('#'): continue
                dest=(p.parent/target.split('#')[0]).resolve()
                assert dest.exists(), f'Broken local link in {rel}: {target}'
                if dest.is_file(): assert dest in files, f'Link to excluded file in {rel}: {target}'
    out=ROOT/'reports'
    for name,item in json.loads((out/'provenance.json').read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==item['sha256'], name
    selected=json.loads((out/'selected-test.json').read_text())
    comparison=json.loads((out/'test-comparison.json').read_text())['fine_tuned']['validation_selected']
    for k in ('precision','recall','f1','tp','fp','tn','fn'): assert selected[k]==comparison[k], k
    assert sum(selected[k] for k in ('tp','fp','tn','fn'))==3000
    print(f'Checked {len(files)} public files: syntax, JSON, links, aggregate hashes, results, exclusions and obvious credential patterns.')
    print('This is a focused repository check, not a guarantee that arbitrary secrets cannot exist.')


if __name__ == '__main__':
    main()
