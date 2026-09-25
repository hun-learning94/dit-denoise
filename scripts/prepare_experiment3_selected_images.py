"""Freeze seed-0 generation tasks from the current selected endpoint table."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'reports/experiment3'
OUT=ROOT/'reports/2026-09-13/selected_endpoint_images'
SPE={'celeba64':5086,'ffhq64':1875,'aahq64':480,'stl10_64':3265,'afhq_v2_64':447,'icartoonface64':12147}

def main():
    selections=json.loads((HERE/'tables/endpoint_selections.json').read_text())
    rows=[];pending=[]
    available=[]
    for name in ['aligned200','reverse_budget200']:
        available+=json.loads((ROOT/f'reports/2026-09-11/{name}_evaluation_manifest.json').read_text())
    for r in selections:
        if r['step'] is None:pending.append(dict(source=r['source'],target=r['target'],reason='No selected paired-FID endpoint'));continue
        source,target=r['source'],r['target'];step=r['step']
        stem=f"primary_20m_{source}_to_{target}_step{step}_generation_seed0"
        legacy=f"primary_20m_{source}_to_{target}_epoch{r['target_epoch']:g}_generation_seed0"
        if all((HERE/'figures'/f'{legacy}.{ext}').exists() for ext in ['png','pdf','npz','json']):continue
        matches=[x for x in available if x['scale']=='20m' and x.get('mode','transfer')=='transfer' and x['seed']==0 and x['target']==target and x['adaptation_updates']==step and (x.get('source') or Path(x['checkpoint']).parts[2].split('_to_')[0])==source]
        assert len(matches)==1,(r,matches)
        adapted=matches[0]['checkpoint']
        source_checkpoint=f'runs/experiment3_20m/{source}/seed0/scratch/checkpoints/ckpt_step{400*SPE[source]}.pt'
        assert (ROOT/adapted).is_file() and (ROOT/source_checkpoint).is_file()
        rows.append(dict(source=source,target=target,seed=0,step=step,selection=r,
                         source_checkpoint=source_checkpoint,adapted=adapted,
                         cube_dir=f'runs/experiment3_selected_images_20m/{source}_to_{target}/seed0/step{step}',
                         output=f'reports/experiment3/figures/{stem}.png'))
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json';text=json.dumps(rows,indent=2)+'\n'
    if path.exists():assert path.read_text()==text,'Do not change an active generation manifest'
    else:path.write_text(text)
    (OUT/'pending.json').write_text(json.dumps(pending,indent=2)+'\n')
    print(json.dumps(dict(tasks=len(rows),directions=[(r['source'],r['target'],r['step']) for r in rows],unselected=pending),indent=2))

if __name__=='__main__':main()
