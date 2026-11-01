"""Classify an English comment locally on Apple Silicon."""
import argparse
import hashlib
import json
from pathlib import Path
import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comment')
    args=parser.parse_args()
    root=Path(snapshot_download('dharshannn/CommentGuard-Qwen3-4B-MLX-LoRA',
                                revision='ef21f5d40dcfb7fb7268b6f3101711281422a2d0'))
    cfg=json.loads((root/'classification_config.json').read_text())
    assert hashlib.sha256((root/'adapters.safetensors').read_bytes()).hexdigest()==cfg['adapter_sha256']
    base=snapshot_download('mlx-community/Qwen3-4B-Instruct-2507-4bit',revision=cfg['model_revision'])
    model,tok=load(base,adapter_path=str(root));model.eval()
    def encode(text):
        prompt=tok.apply_chat_template([
            {'role':'system','content':cfg['system_prompt']},
            {'role':'user','content':'Comment:\n'+text}],tokenize=False,add_generation_prompt=True)+cfg['wrapper']
        return tok.encode(prompt,add_special_tokens=False)
    text=args.comment;ids=encode(text);tokens=tok.encode(text,add_special_tokens=False);truncated=False
    while len(ids)>cfg['context']-1:
        truncated=True;tokens=tokens[:max(0,len(tokens)-(len(ids)-(cfg['context']-1)))];ids=encode(tok.decode(tokens))
    label_ids=[tok.encode(s,add_special_tokens=False) for s in ['0','1']]
    assert label_ids==[[15],[16]]
    logits=model(mx.array([ids]))[0,-1,:].astype(mx.float32)
    score=mx.softmax(logits[mx.array([15,16])])[1].item()
    print(json.dumps({'label':int(score>=cfg['threshold']),'toxic_score':score,
                     'threshold':cfg['threshold'],'truncated':truncated}))


if __name__=='__main__':main()
