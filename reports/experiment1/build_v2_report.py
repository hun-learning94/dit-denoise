"""Rebuild the completed v2 study from metric artifacts and training provenance."""
from pathlib import Path
import csv,json,re,math,collections,sys
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from reports.experiment1.build_report_assets64 import records,mse_at,plot_trajectories,plot_efficiency
FIG=HERE/'figures_v2';FIG.mkdir(exist_ok=True)
GRID=json.loads((HERE/'experiment1_v2_manifest.json').read_text())
COLORS={'DiT':'#2463a6','DPD5 H1':'#21a179','DPD5 H8':'#80bd53','Grouped DPD6 H1':'#843c99','Grouped DPD6 H8':'#d05b9c'}
METRICS=[('vp_velocity_loss','vp_velocity_loss','VP velocity loss'),('denoising','log_sigma_auc_mse_raw','Denoising AUC'),('fid50k','fid','FID-50k')]

def label(r):
 return f'DiT F{r["width"]}' if r['model']=='dit' else f'{"DPD5" if r["model"]=="dpd5" else "Grouped DPD6"} H{r["H"]} W{r["width"]}'
def family(r):return 'DiT' if r['model']=='dit' else f'{"DPD5" if r["model"]=="dpd5" else "Grouped DPD6"} H{r["H"]}'
def read_metrics(r):
 prefix=Path(r['run_dir']).name;root=ROOT/('evals/experiment1_celeba64' if r['reuse'] else 'evals/experiment1_v2');out={}
 for k,field,_ in METRICS:
  fn='result.json' if k=='fid50k' else 'summary.json'
  for p in (root/k).glob(prefix+'_epoch*/'+fn):
   e=int(re.search(r'epoch(\d+)',p.parent.name)[1]);o=json.loads(p.read_text());assert math.isfinite(float(o[field]))
   if k=='fid50k':assert o['generated_images']==50000
   assert k not in out.get(e,{}),(r,e,k);out.setdefault(e,{})[k]=o
 return out

def training():
 allocations=list(csv.DictReader((HERE/'training_records/v2_allocations.tsv').open(),delimiter='|'))
 reused=list(csv.DictReader((HERE/'training_records/reused_allocations.tsv').open(),delimiter='|'))
 devices=[json.loads(p.read_text()) for p in (HERE/'v2_device_records').glob('*.json')]
 result=[]
 for r in GRID:
  task=r['task'];row=dict(task=task,model=label(r));gpu_times=collections.Counter()
  if not r['reuse']:
   parts=[a for a in allocations if int(a['JobID'].split('_')[-1])==task]
   successful=[a for a in parts if a['State'] in ['REQUEUED','COMPLETED']]
   assert any(a['State']=='COMPLETED' for a in parts)
   for a in successful:
    names={d['gpu'] for d in devices if d['task']==task and d['node']==a['NodeList']}
    assert len(names)==1,(task,a,names)
    gpu_times[next(iter(names))]+=int(a['ElapsedRaw'])
   seconds=sum(int(a['ElapsedRaw']) for a in parts)
   row.update(active_hours=seconds/3600,effective_iterations_per_second=2034400/seconds,
     gpus='; '.join(f'{g.replace("NVIDIA ","")}: {v/3600:.2f} h' for g,v in gpu_times.items()),method='all training allocations; includes checkpoint/sample/startup and failed-start overhead; excludes queue waits')
  elif task in [0,1,6]:
   idx={0:0,1:1,6:3}[task];parts=[a for a in reused if a['JobID']==f'53702391_{idx}'];seconds=sum(int(a['ElapsedRaw']) for a in parts)
   row.update(active_hours=seconds/3600,effective_iterations_per_second=2034400/seconds,gpus='GeForce RTX 2080 Ti (retained log; earlier allocations lack device records)',method='all original training allocations; historical per-GPU breakdown unavailable')
  else:
   # Stop the shared allocation clock at this model's durable completion.
   done=(ROOT/r['run_dir']/'checkpoints/ckpt_step2034400.pt').stat().st_mtime
   parts=list(csv.DictReader((HERE/'training_records/grouped_reused_allocations.tsv').open(),delimiter='|'))
   seconds=sum(max(0,min(done,datetime.fromisoformat(a['End']).timestamp())-datetime.fromisoformat(a['Start']).timestamp()) for a in parts)
   seconds+=sum(int(a['ElapsedRaw']) for a in reused if a['JobID']=='53702526_0')
   row.update(active_hours=seconds/3600,effective_iterations_per_second=2034400/seconds,gpus='H200 (shared with H14/H18; fused BF16 implementation)',method='approximate per-model active span: shared allocation intervals truncated at epoch-400 checkpoint mtime, plus failed initial start; not exclusive GPU-hours')
  if not r['reuse']:log=ROOT/f'slurm-exp1-v2-{54645329 if task in [2,3,4,5] else 54504803}_{task}.out'
  elif task in [0,1,6]:log=ROOT/f'slurm-exp1-64-e400-53702391_{ {0:0,1:1,6:3}[task]}.out'
  else:log=ROOT/'slurm-exp1-64-g6-h200-53775977-h8.out'
  text=log.read_text();matches=re.findall(r'\[train\] step\s+(\d+)/\d+[^\n]*?([\d.]+) it/s',text)
  resumes=re.findall(r'at step=(\d+)',text);start=int(resumes[-1]) if resumes else 0
  if matches:
   step,rate=matches[-1];row.update(last_segment_start=start,last_segment_end=int(step),last_segment_iterations_per_second=float(rate),last_segment_estimated_hours=(int(step)-start)/float(rate)/3600,log=str(log.relative_to(ROOT)))
  result.append(row)
 (HERE/'training_records/v2_training_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def main():
 # Retained entry point: regenerate the unified report rather than a separate v2 report.
 from reports.experiment1.build_combined_report import main as combined_main
 combined_main()

if __name__=='__main__':main()
