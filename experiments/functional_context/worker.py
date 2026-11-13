"""Class-token-safe JiT context sweeps, with durable progress each inference batch."""
import argparse
import fcntl
import gc
import os
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .protocol import H, read, write, save, sha, ranks

M = None
MH = None
from experiments.attention_context_transfer.donor_models import Model
from experiments.attention_context_transfer.jit_transfer import protected_rows
from experiments.attention_context_transfer.imagenet_fields import clean
from experiments.attention_context_transfer.protocol import seed_for
STOP = False
RANK = ranks()


def stop(*_):
    global STOP
    STOP = True


def observation(rec, sigma):
    x = clean(rec)
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(
        seed_for(M['noise_seed_namespace'], rec['image_id'], 0)))
    t = 1/(1+sigma)
    return t*x+(1-t)*z, t


def target_pixels(pred, q):
    q = torch.as_tensor(q, dtype=torch.long, device=pred.device)
    offsets = torch.arange(16, device=pred.device)
    y = (q//16)[:, None, None]*16+offsets[None, :, None]
    x = (q%16)[:, None, None]*16+offsets[None, None, :]
    index = (y*256+x).reshape(len(q), 1, 256).expand(-1, 3, -1)
    return pred.flatten(2).gather(2, index).reshape(len(q), -1)


def masks(q, k):
    return torch.as_tensor(RANK[np.asarray(q)] < np.asarray(k)[:, None], device='cuda')


def preflight(model, name):
    dest = H/name;dest.mkdir(exist_ok=True)
    if (dest/'PREFLIGHT.json').exists():
        gate = read(dest/'PREFLIGHT.json');assert gate['manifest_sha256'] == MH;return
    # Include single-token contexts and protect all class tokens at every layer.
    queries = [0, 15, 85, 136, 240, 255];ks = [1, 2, 9, 32, 128, 255]
    obs = [observation(rec, sigma) for rec, sigma in zip(M['calibration'][:6], [.02, 2, 8, 24, 32, 40])]
    xx = torch.stack([x for x,t in obs]).cuda()
    tt = torch.tensor([t for x,t in obs], device='cuda')
    ll = torch.tensor([r['class_id'] for r in M['calibration'][:6]], device='cuda')
    keep = masks(queries, ks)
    trace = []
    def trace_hook(block, args, output):
        trace.append(output[protected_rows(keep, output.shape[1]-256)].detach().clone())
    handles = [b.register_forward_hook(trace_hook) for b in model.model.blocks]
    with torch.inference_mode():
        dense = model(xx, tt, ll);trace.clear()
        full = model(xx, tt, ll, torch.ones_like(keep));trace.clear()
        masked = model(xx, tt, ll, keep);first = trace[:];trace.clear()
        pixel_keep = keep.reshape(6, 1, 16, 16).repeat_interleave(16, 2).repeat_interleave(16, 3)
        changed = model(torch.where(pixel_keep, xx, torch.randn_like(xx)*3), tt, ll, keep)
        second = trace[:];trace.clear()
        for handle in handles:handle.remove()
        single = torch.cat([model(xx[i:i+1], tt[i:i+1], ll[i:i+1], keep[i:i+1]) for i in range(6)])
    fullerr = float((full-dense).abs().max())
    outside = float(((masked-changed)*pixel_keep).abs().max())
    hidden = [float((a-b).abs().max()) for a,b in zip(first, second)]
    batcherr = float((single-masked).abs().max())
    assert fullerr < 2e-5 and outside < 2e-6 and max(hidden) < 2e-6
    assert batcherr < 2e-4 and float((masked-dense).abs().max()) > 1e-4
    yy = xx[:1].detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(yy, tt[:1], ll[:1], keep[:1])
        value = target_pixels(out, queries[:1]).square().sum()
        grad, = torch.autograd.grad(value, yy)
    graderr = float((grad*~pixel_keep[:1]).abs().max());assert graderr == 0
    # Check patch extraction against direct RGB slicing.
    extracted = target_pixels(dense, queries)
    direct = torch.stack([dense[i, :, q//16*16:(q//16+1)*16, q%16*16:(q%16+1)*16].reshape(-1)
                          for i,q in enumerate(queries)])
    assert torch.equal(extracted, direct)
    del full,dense,masked,first,second,single,changed,grad,yy,out,value,extracted,direct
    gc.collect();torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory;bench=[]
    for b in [32,64,128,256,512,768,1024]:
        if bench and bench[-1]['peak']/bench[-1]['batch']*b > .85*total:break
        xb=tb=lb=kb=pred=None
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            ids = torch.arange(b, device='cuda')%6
            xb,tb,lb,kb = xx[ids],tt[ids],ll[ids],keep[ids]
            with torch.inference_mode():
                pred = model(xb,tb,lb,kb);del pred;pred=None;torch.cuda.synchronize()
                start=time.monotonic();pred=model(xb,tb,lb,kb);torch.cuda.synchronize()
                elapsed=time.monotonic()-start
                assert torch.isfinite(pred).all()
            peak=torch.cuda.max_memory_reserved()
            bench.append(dict(batch=b,seconds=elapsed,peak=peak,contexts_per_second=b/elapsed,
                              admitted=peak < .8*total))
            print('BENCHMARK',name,bench[-1],flush=True)
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            del xb,tb,lb,kb,pred;gc.collect();torch.cuda.empty_cache()
        if STOP:raise SystemExit(75)
    good=[v for v in bench if v['admitted']];assert good
    fastest=max(v['contexts_per_second'] for v in good)
    chosen=max((v for v in good if v['contexts_per_second'] >= .9*fastest),key=lambda v:v['batch'])
    # A small separate calibration diagnostic, with no tolerance selection.
    calibration=[]
    with torch.inference_mode():
        for sigma in [.02,1.,49.]:
            x,t=observation(M['calibration'][0],sigma);x=x[None].cuda()
            t=torch.tensor([t],device='cuda');label=ll[:1]
            dense=model(x,t,label);qs=[85]*6;kk=[1,9,32,96,192,256]
            pred=model(x.expand(6,-1,-1,-1),t.expand(6),label.expand(6),masks(qs,kk))
            delta=(target_pixels(pred,qs)-target_pixels(dense,[85])).square().mean(-1).cpu().numpy()
            assert np.isfinite(delta).all() and delta[-1] < 1e-8
            calibration.append(dict(sigma=sigma,budgets=kk,prediction_mse=delta.tolist()))
    write(dest/'PREFLIGHT.json',dict(passed=True,manifest_sha256=MH,at=time.time(),
          gpu=torch.cuda.get_device_name(),model=model.receipt,full_context_max_abs=fullerr,
          outside_output_max_abs=outside,hidden_max_abs_by_block=hidden,
          outside_gradient_max_abs=graderr,batch_single_max_abs=batcherr,
          class_queries_protected=True,single_patch_context_checked=True,
          patch_extraction_exact=True,benchmarks=bench,selected=chosen,
          calibration=calibration,tolerances_unchanged=M['tolerances'],
          estimated_gpu_hours=256*8*len(M['sigmas'])*255/chosen['contexts_per_second']/3600,
          estimated_wall_hours_two_workers=256*8*len(M['sigmas'])*255/chosen['contexts_per_second']/7200))
    print('PREFLIGHT_PASS',name,chosen,flush=True)


def run(model, name, shard):
    gate=read(H/name/'PREFLIGHT.json');assert gate['passed'] and gate['manifest_sha256']==MH
    batch=gate['selected']['batch'];work=Path(M['work'])/name;work.mkdir(parents=True,exist_ok=True)
    ids=list(range(shard,256,2));started=time.monotonic();completed=0
    qindex=np.repeat(np.arange(8),255);budgets=np.tile(np.arange(1,256),8)
    with torch.inference_mode():
        for si,sigma in enumerate(M['sigmas']):
            for ii in ids:
                path=work/f'sigma{si:02d}_image{ii:03d}.npz'
                state=None
                if path.exists():
                    with np.load(path) as z:state={k:z[k] for k in z.files}
                    assert str(state['manifest_sha256'])==MH
                    if state['done'].all():completed+=1;continue
                rec=M['records'][ii];queries=np.array(rec['targets']);x,t=observation(rec,sigma)
                x=x[None].cuda();tt=torch.tensor([t],device='cuda');ll=torch.tensor([rec['class_id']],device='cuda')
                truth=clean(rec)[None].cuda()
                dense=model(x,tt,ll)
                dense_p=target_pixels(dense.expand(8,-1,-1,-1),queries)
                truth_p=target_pixels(truth.expand(8,-1,-1,-1),queries)
                dense_loss=(dense_p-truth_p).square().mean(-1)
                if state is None:
                    delta=np.full((8,256),np.nan,np.float32);loss=delta.copy();done=np.zeros((8,256),bool)
                    delta[:,-1]=0;loss[:,-1]=dense_loss.cpu().numpy();done[:,-1]=True
                else:
                    delta,loss,done=state['discrepancy'],state['clean_mse'],state['done']
                    np.testing.assert_allclose(state['dense_patch'],dense_p.cpu().numpy(),atol=2e-5,rtol=2e-5)
                todo=np.flatnonzero(~done[qindex,budgets-1])
                for first in range(0,len(todo),batch):
                    chunk=todo[first:first+batch];qi=qindex[chunk];kk=budgets[chunk];q=queries[qi];n=len(chunk)
                    pred=model(x.expand(n,-1,-1,-1),tt.expand(n),ll.expand(n),masks(q,kk))
                    patch=target_pixels(pred,q)
                    dd=(patch-dense_p[torch.as_tensor(qi,device='cuda')]).square().mean(-1).cpu().numpy()
                    ee=(patch-truth_p[torch.as_tensor(qi,device='cuda')]).square().mean(-1).cpu().numpy()
                    assert np.isfinite(dd).all() and np.isfinite(ee).all()
                    delta[qi,kk-1]=dd;loss[qi,kk-1]=ee;done[qi,kk-1]=True
                    save(path,manifest_sha256=MH,discrepancy=delta,clean_mse=loss,done=done,
                         dense_mse=dense_loss.cpu().numpy(),dense_patch=dense_p.cpu().numpy(),
                         queries=queries,image_index=ii,sigma=sigma,class_id=rec['class_id'])
                    del pred,patch
                    write(H/name/f'PROGRESS_{shard}.json',dict(at=time.time(),sigma=sigma,
                          image_index=ii,completed_image_times=completed,total_image_times=128*len(M['sigmas']),
                          masks_in_image_time=int(done.sum()),batch=batch,job=os.environ.get('SLURM_JOB_ID'),
                          seconds_this_allocation=time.monotonic()-started))
                    if STOP:raise SystemExit(75)
                assert done.all();completed+=1
                del dense,dense_p,truth_p,x,truth
                print('PROGRESS',name,shard,completed,'/',128*len(M['sigmas']),flush=True)
                if STOP:raise SystemExit(75)
    write(H/name/f'COMPLETE_{shard}.json',dict(passed=True,manifest_sha256=MH,at=time.time(),
          image_times=completed,work=str(work),shard=shard))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run']);p.add_argument('--index',type=int,required=True);a=p.parse_args()
    global M, MH
    M = read(H/'manifest.json'); MH = sha(H/'manifest.json')
    for sig in [signal.SIGUSR1,signal.SIGTERM]:signal.signal(sig,stop)
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    for path,expected in M['source_hashes'].items():assert sha(path)==expected,path
    index=a.index if a.stage=='preflight' else a.index//2
    name=M['models'][index];dest=H/name;dest.mkdir(exist_ok=True)
    with (dest/f'{a.stage}_{a.index}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=Model(name,M['design']);model.model.float()
        if a.stage=='preflight':preflight(model,name)
        else:run(model,name,a.index%2)
if __name__=='__main__':main()
