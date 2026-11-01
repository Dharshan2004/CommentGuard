"""Publish an allowlisted, aggregate-only snapshot from local experiment outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'test-baseline-comparison/metrics.json': 'test-comparison.json',
    'test-main5000-step4000/metrics.json': 'selected-test.json',
    'iterative-selection/metrics.json': 'validation-comparison.json',
    'iterative-selection/SELECTED_CONFIG.json': 'frozen-model.json',
    'test-base-labels/frozen_config.json': 'frozen-base.json',
    'test-logistic-main5000/frozen_config.json': 'frozen-logistic.json',
}


def main():
    out = ROOT/'reports';out.mkdir(exist_ok=True)
    provenance = {}
    for source, name in FILES.items():
        p = ROOT/'results'/source
        data = p.read_bytes()
        json.loads(data)
        assert b'/Users/' not in data and b'/home/' not in data
        (out/name).write_bytes(data)
        provenance[name] = dict(source='results/'+source, sha256=hashlib.sha256(data).hexdigest())
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Exported {len(FILES)} aggregate/config snapshots; no examples or weights.')


if __name__ == '__main__':
    main()
