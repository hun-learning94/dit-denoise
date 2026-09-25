#!/usr/bin/env python3
"""Build 64×64 Experiment 1 trajectories and parameter-efficiency panels."""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; EVAL=ROOT/"evals/experiment1_celeba64"
OUT=Path(__file__).resolve().parent; FIG=OUT/"figures64"; REFERENCE_EPOCH=400
MODELS=[
 ("celeba64_unet_xpred_vp_vloss_b32","U-Net","U-Net",10_079_590),
 ("celeba64_dit_p4_xpred_vp_vloss_f640_low_b32","DiT F640","DiT",8_639_536),
 ("celeba64_dit_p4_xpred_vp_vloss_f1024_matched_b32","DiT F1024","DiT",10_215_472),
 ("celeba64_dit_p4_xpred_vp_vloss_f1280_high_b32","DiT F1280","DiT",11_266_096),
 ("celeba64_dpd5_p4_xpred_vp_vloss_r1024_h8_d32_low_b32","DPD5 H8","DPD5",8_642_672),
 ("celeba64_dpd5_p4_xpred_vp_vloss_r1024_h14_d32_matched_b32","DPD5 H14","DPD5",10_221_728),
 ("celeba64_dpd5_p4_xpred_vp_vloss_r1024_h18_d32_high_b32","DPD5 H18","DPD5",11_274_432),
 ("celeba64_grouped_dpd6_p4_xpred_vp_vloss_r1024_h8_d32_low_b32","gDPD6 H8","Grouped DPD6",8_652_848),
 ("celeba64_grouped_dpd6_p4_xpred_vp_vloss_r1024_h14_d32_matched_b32","gDPD6 H14","Grouped DPD6",10_239_536),
 ("celeba64_grouped_dpd6_p4_xpred_vp_vloss_r1024_h18_d32_high_b32","gDPD6 H18","Grouped DPD6",11_297_328),
 ("celeba64_dpd6_p4_xpred_vp_vloss_both_r1024_h1_b32","Original DPD6","Original DPD6",7_587_888)]
COLORS={"U-Net":"#7f7f7f","DiT":"#1f77b4","DPD4":"#d62728","DPD5":"#2ca02c","Grouped DPD6":"#9467bd","Original DPD6":"#ff7f0e"}
MARKERS=("o","s","^","D","P","X","v","<",">","h","*","8","p","H","d")

def ep(path): return int(re.search(r"epoch(\d+)",path.parent.name).group(1))
def records(prefix):
 out={}
 for kind,name in (("denoising","summary.json"),("vp_velocity_loss","summary.json"),("fid50k","result.json")):
  for path in (EVAL/kind).glob(f"{prefix}_epoch*/{name}"): out.setdefault(ep(path),{})[kind]=json.loads(path.read_text())
 return out
def mse_at(d,sigma):
 i=min(range(len(d["sampler_sigmas"])),key=lambda j:abs(d["sampler_sigmas"][j]-sigma)); return float(d["sampler_raw_mse"][i])

def plot_trajectories(models,allr,colors,figdir):
 """All models on the same three metric axes; missing epochs break curves."""
 fig,axs=plt.subplots(1,3,figsize=(18.5,5.2),constrained_layout=True)
 for i,(p,label,fam,_) in enumerate(models):
  r=allr[p]
  series=[[(e,r.get(e,{}).get(k,{}).get(field,float("nan"))) for e in range(25,401,25)] for k,field in (("vp_velocity_loss","vp_velocity_loss"),("denoising","log_sigma_auc_mse_raw"),("fid50k","fid"))]
  for ax,v in zip(axs,series):
   if v: ax.plot(*zip(*v),label=label,color=colors[fam],lw=1.4,marker=MARKERS[i],ms=3,ls=["-","--",":"][i%3])
 for ax,title,y in zip(axs,("Held-out VP velocity loss","Denoising AUC","FID-50k"),("VP velocity loss","Log-$\\sigma$ AUC MSE","FID-50k")):
  ax.set(title=title,xlabel="Training epoch",ylabel=y,xlim=(20,405)); ax.grid(alpha=.22)
 axs[2].legend(fontsize=6.7,ncol=2)
 for ext in ["png","pdf"]: fig.savefig(figdir/f"metric_trajectories.{ext}",dpi=220)
 plt.close(fig)

def plot_efficiency(rows,colors,figdir,title="CelebA64: epoch 400, EMA, all three paired metrics",model_order=None):
 """Unconnected epoch-matched scatter points with a common model legend."""
 ref=next(x for x in rows if x["model"]=="DiT F1024"); metrics=(("mse_low","Relative raw MSE, low noise"),("mse_mid","Relative raw MSE, mid noise"),("mse_high","Relative raw MSE, high noise"),("denoising_auc","Denoising AUC"),("vp_velocity_loss","Held-out VP velocity loss"),("fid","FID-50k"))
 fig,axs=plt.subplots(2,3,figsize=(13.5,8.2),constrained_layout=True)
 marker_by_model={name:MARKERS[i] for i,name in enumerate(model_order or [r['model'] for r in rows])}
 for ax,(key,axis_title) in zip(axs.ravel(),metrics):
  for i,x in enumerate(rows):
   val=float(x[key])/float(ref[key]) if key.startswith("mse_") else float(x[key]); ax.scatter(x["parameters"]/1e6,val,s=65,color=colors[x["family"]],marker=marker_by_model[x['model']],label=x["model"])
  ax.set(title=axis_title,xlabel="Trainable parameters (M)"); ax.grid(alpha=.22)
 fig.suptitle(title,fontsize=12)
 handles,labels=axs[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.075),ncol=4,frameon=False,fontsize=9)
 for ext in ["png","pdf"]: fig.savefig(figdir/f"parameter_efficiency_epoch400.{ext}",dpi=220,bbox_inches="tight")
 fig.savefig(figdir/"parameter_efficiency_latest.png",dpi=220,bbox_inches="tight"); plt.close(fig)

def main():
 FIG.mkdir(exist_ok=True); allr={p:records(p) for p,*_ in MODELS}
 plot_trajectories(MODELS,allr,COLORS,FIG)
 rows=[]
 for p,label,fam,params in MODELS:
  r=allr[p]; common=[e for e,x in r.items() if e<=400 and all(k in x for k in ("denoising","vp_velocity_loss","fid50k"))]
  if not common: continue
  assert REFERENCE_EPOCH in common, f"Missing epoch-{REFERENCE_EPOCH} metrics: {label}"
  e=REFERENCE_EPOCH; d,v,f=r[e]["denoising"],r[e]["vp_velocity_loss"],r[e]["fid50k"]
  rows.append(dict(model=label,family=fam,parameters=params,epoch=e,mse_low=mse_at(d,.1),mse_mid=mse_at(d,1),mse_high=mse_at(d,4),denoising_auc=d["log_sigma_auc_mse_raw"],vp_velocity_loss=v["vp_velocity_loss"],fid=f["fid"]))
 with (OUT/"preliminary_metrics64.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 plot_efficiency(rows,COLORS,FIG)
 print(f"wrote {len(rows)} paired 64x64 records")
if __name__=="__main__":
 import sys
 sys.path.insert(0,str(ROOT))
 from reports.experiment1.build_combined_report import main as combined_main
 combined_main()
