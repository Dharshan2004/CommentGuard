"""Wait for a specified Studio experiment, then evaluate its best-loss checkpoint.

Defaults preserve the original dropout0 control run. For another arm, supply
--run-id, --name, --progress-name, --label, and --next-action explicitly.
Never starts or restarts training, and never evaluates the test set.
"""
import argparse
import datetime
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='job_20261101_114901_3a1542d4'
NAME='epochs2-dropout0-best'
DB=Path.home()/'.unsloth/studio/studio.db'
OUT=ROOT/'results'/('validation-'+NAME)
MODEL=str(Path.home()/'.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b')
PROGRESS=ROOT/'results/iterative-selection/control_followup.json'
LABEL='Two-epoch dropout0 control'
NEXT_ACTION='Start matched fresh-base two-epoch dropout0.1 run in Studio'

def configure():
    global RUN, NAME, OUT, PROGRESS, LABEL, NEXT_ACTION
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',default=RUN)
    ap.add_argument('--name',default=NAME)
    ap.add_argument('--progress-name',default='control_followup')
    ap.add_argument('--label',default=LABEL)
    ap.add_argument('--next-action',default=NEXT_ACTION)
    args=ap.parse_args()
    for name in (args.name,args.progress_name):
        if not name or Path(name).name!=name or name in ('.','..'):
            ap.error('Names must be single directory/file stems')
    if args.run_id!=RUN and (args.name==NAME or args.progress_name=='control_followup'):
        ap.error('A different run requires its own --name and --progress-name')
    RUN,NAME,LABEL,NEXT_ACTION=args.run_id,args.name,args.label,args.next_action
    OUT=ROOT/'results'/('validation-'+NAME)
    PROGRESS=ROOT/'results/iterative-selection'/(args.progress_name+'.json')

def record(**fields):
    fields['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    PROGRESS.write_text(json.dumps(fields,indent=2)+'\n')

def command(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],check=True)

def main():
    last=None
    while True:
        c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
        row=c.execute('select status,final_step,total_steps,error_message from training_runs where id=?',(RUN,)).fetchone()
        ev=c.execute('select step,eval_loss from training_metrics where run_id=? and eval_loss is not null order by step',(RUN,)).fetchall()
        c.close()
        if not row:raise RuntimeError('Run missing; refusing restart')
        status,step,total,error=row
        record(phase='waiting_for_training',run_id=RUN,training_status=status,step=step,total=total,evaluations=ev)
        key=(status,step//500,len(ev))
        if key!=last:
            print(json.dumps(dict(status=status,step=step,total=total,last_eval=ev[-1] if ev else None)),flush=True);last=key
        if status=='completed':break
        if status not in ('running','training','saving'):
            raise RuntimeError(f'Unexpected terminal state {status}: {error}')
        time.sleep(30)
    record(phase='staging_best_checkpoint',run_id=RUN)
    command('stage_best_checkpoint.py','--run-id',RUN,'--name',NAME)
    record(phase='scoring_validation',run_id=RUN)
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Existing scoring manifest: inspect before rerun')
    command('score_labels.py','--model-path',MODEL,'--adapters',NAME,'--output',str(OUT))
    record(phase='threshold_selection',run_id=RUN)
    command('select_threshold.py','--scores',str(OUT))
    result=json.loads((OUT/'threshold_results.json').read_text())[NAME]
    with (ROOT.parent/'CommentGuard-WORKLOG.md').open('a') as f:
        f.write('\n### '+LABEL+' completed and scored\n\nRun '+RUN+' completed. Selected minimum-token-loss checkpoint using recorded rule and verified native weights. Scored all 1,000 validation rows with unchanged protocol; fixed 0.5 metrics '+json.dumps(result['fixed_0_5'])+'; five-fold metrics '+json.dumps(result['out_of_fold'])+'; AP '+str(result['average_precision'])+'. Full-validation threshold '+str(result['full_validation_threshold'])+' is development-only. Next action: '+NEXT_ACTION+'. Test remains unevaluated.\n')
    record(phase='evaluation_completed',run_id=RUN,result=result,next_action=NEXT_ACTION)
    print(LABEL+' EVALUATION COMPLETE. Next action: '+NEXT_ACTION,flush=True)

if __name__=='__main__':
    configure()
    try:main()
    except Exception as exc:
        record(phase='followup_error',run_id=RUN,error=str(exc));raise
