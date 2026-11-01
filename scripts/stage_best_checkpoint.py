"""Stage a completed Studio run's minimum-validation-loss native MLX adapter.

Never stages a running run. Tie-break by earlier step. Does not score any data.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DB=Path.home()/'.unsloth/studio/studio.db'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--name',required=True)
    args=ap.parse_args()
    assert Path(args.name).name==args.name and args.name not in ('.','..')
    con=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    run=con.execute('select status,output_dir,total_steps,final_step from training_runs where id=?',(args.run_id,)).fetchone()
    if not run or run[0]!='completed': raise RuntimeError('Run must be confirmed completed before checkpoint selection')
    status,directory,total,final=run
    assert total==final
    source=Path(directory)
    evals=con.execute('select step,eval_loss from training_metrics where run_id=? and eval_loss is not null order by step',(args.run_id,)).fetchall()
    con.close()
    assert evals and evals[-1][0]==final
    step,loss=min(evals,key=lambda pair:(pair[1],pair[0]))
    checkpoint=source/f'checkpoint-{step}'
    state=json.loads((source/f'checkpoint-{final}'/'trainer_state.json').read_text())
    assert evals==[(r['step'],r['eval_loss']) for r in state['log_history'] if 'eval_loss' in r]
    config=json.loads((checkpoint/'adapter_config.json').read_text())
    final_config=json.loads((source/'adapter_config.json').read_text())
    revision=config.get('base_model_revision',config.get('base_model_commit_hash'))
    assert revision==final_config.get('base_model_revision',final_config.get('base_model_commit_hash'))
    dst=ROOT/'adapters'/args.name
    files={n:checkpoint/n for n in ('adapter_config.json','adapters.safetensors')}
    files.update({n:source/n for n in ('chat_template.jinja','tokenizer.json','tokenizer_config.json')})
    for n,p in files.items():
        assert p.is_file()
        if (dst/n).exists(): assert sha(dst/n)==sha(p), 'Refusing to overwrite differing adapter'
    dst.mkdir(exist_ok=True)
    for n,p in files.items():
        shutil.copy2(p,dst/n);assert sha(p)==sha(dst/n)
    evidence=dict(run_id=args.run_id,selection='minimum recorded validation loss; ties earlier step',
        source_checkpoint=str(checkpoint),step=step,eval_loss=loss,final_eval_loss=evals[-1][1],
        base_revision=revision,files_sha256={n:sha(dst/n) for n in files})
    (dst/'export_metadata.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
