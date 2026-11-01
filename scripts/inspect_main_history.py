"""Extract saved main-run loss history without evaluating any dataset."""
import csv
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'.unsloth/studio/outputs/mlx-community_Qwen3-4B-Instruct-2507-4bit_1793487135'
OUT=ROOT/'results/validation-main5000-001'

def main():
    state_path=BASE/'checkpoint-5000/trainer_state.json'
    state=json.loads(state_path.read_text())
    conn=sqlite3.connect(f'file:{Path.home()}/.unsloth/studio/studio.db?mode=ro',uri=True)
    run=conn.execute('select id,status,total_steps,final_step from training_runs where output_dir=?',(str(BASE),)).fetchone()
    assert run[1:] == ('completed',5000,5000)
    evals=conn.execute('select step,eval_loss from training_metrics where run_id=? and eval_loss is not null order by step',(run[0],)).fetchall()
    conn.close()
    assert evals == [(r['step'],r['eval_loss']) for r in state['log_history'] if 'eval_loss' in r]
    train=state['train_loss_history']; assert len(train)==5000
    rows=[dict(step=step,epoch=step/5000,eval_loss=loss,
        preceding_500_step_train_mean=statistics.mean(train[step-500:step])) for step,loss in evals]
    with (OUT/'training_history.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    best=min(rows,key=lambda r:r['eval_loss'])
    evidence=dict(source_state=str(state_path),source_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
        run_id=run[0],database_and_checkpoint_eval_history_match=True,best_eval_step=best['step'],
        best_eval_loss=best['eval_loss'],final_eval_loss=rows[-1]['eval_loss'],
        checkpoint_4000_exists=(BASE/'checkpoint-4000/adapters.safetensors').exists(),
        test_evaluated=False)
    (OUT/'training_history_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['# Saved training-history diagnosis','',
        'Read the completed main5000 run from Studio in read-only mode and cross-checked all ten evaluation values against checkpoint-5000/trainer_state.json. No dataset examples were read or scored.', '',
        '| Step | Epoch | Mean training loss over preceding 500 steps | Validation loss |',
        '|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['step']} | {r['epoch']:.1f} | {r['preceding_500_step_train_mean']:.4f} | {r['eval_loss']:.4f} |")
    lines += ['', '## Findings and next experiment', '',
        '- The early training-loss drop includes learning the required response format; it is not a direct classification-accuracy measure.',
        '- After step 1,000, block-mean training loss stays roughly 0.27–0.29. Validation fluctuates, rather than showing sustained deterioration.',
        '- Best recorded validation loss is 0.249804 at step 4,000 (0.8 epoch); final is 0.256519. The temporary rise at step 2,000 subsequently reversed.',
        '- This is insufficient evidence to diagnose overfitting or to claim dropout 0.1 will help. Extra epochs might help or hurt; the trajectory does not guarantee either.',
        '- Next proposed check: score the already-saved step-4,000 checkpoint on validation using the same label-scoring protocol, fixed 0.5 and the same five-fold threshold objective. Selection by token loss does not guarantee higher toxic F1 or AP.',
        '- That checkpoint comparison has NOT been run by this diagnostic. No new training was started. Repeated validation selection remains development; retain the reserved test for a frozen final configuration.', '',
        '![Loss history](training_history.png)']
    (OUT/'TRAINING_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['step'] for r in rows],[r['preceding_500_step_train_mean'] for r in rows],marker='o')
    axes[0].set_title('Training: mean over each 500-step block')
    axes[1].plot([r['step'] for r in rows],[r['eval_loss'] for r in rows],marker='o',color='#d04b34')
    axes[1].scatter([best['step']],[best['eval_loss']],color='black',zorder=3,label='Best recorded: step 4,000')
    axes[1].legend();axes[1].set_title('Validation: recorded evaluation loss')
    for ax in axes: ax.set_xlabel('Optimizer step');ax.set_ylabel('Loss');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training_history.png',dpi=170);plt.close(fig)
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
