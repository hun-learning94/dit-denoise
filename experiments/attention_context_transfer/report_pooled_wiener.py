"""Publish pooled Wiener alongside the unchanged class-specific comparison."""
from pathlib import Path
import csv,json,os,re,subprocess,sys
import numpy as np
from .pooled_wiener import checked_record,get_fields,record,work
from .full_covariance_cpu import read,digest,save_npz


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stability(root):
    from .report_donor_full import estimates
    from .jit_transfer import order
    d=read(root/'design.json');rng=np.random.default_rng(20261109);indices=[]
    for _ in range(2000):
        classes=rng.integers(32,size=32);indices.append(np.concatenate([c*2+rng.integers(2,size=2) for c in classes]))
    rows=[]
    for name in d['models']:
        r=checked_record(root,name+'/STABILITY_COMPLETE.json')
        with np.load(r['path']) as z:
            excess=(z['losses']-z['independent_dense'][...,None,None]).mean((1,3))
        mean,ci=estimates(excess[...,1]-excess[...,0],indices)
        for ti,t in enumerate(d['primary_times']):
            for ki,k in enumerate(d['budgets']):rows.append(dict(recipient=name,t=t,budget=k,mean_50k_minus_25k=float(mean[ti,ki]),low=float(ci[0,ti,ki]),high=float(ci[1,ti,ki])))
    table(root/'sample_size_loss_comparison.csv',rows)
    f25=get_fields(root,25000);f50=get_fields(root,50000);rows=[]
    for t in d['primary_times']:
        for q in range(256):
            a=order(f25['score'][list(f25['times']).index(t),q],q);b=order(f50['score'][list(f50['times']).index(t),q],q)
            for k in d['budgets'][1:-1]:
                aa=set(a[9:k]);bb=set(b[9:k]);rows.append(dict(t=t,query=q,budget=k,overlap_beyond_core=len(aa&bb)/(k-9),jaccard_beyond_core=len(aa&bb)/len(aa|bb)))
    table(root/'sample_size_context_comparison.csv',rows)
    (root/'STABILITY.md').write_text('''# 25k versus 50k sample-size check

Both banks use a fixed class-balanced training sample; 25k is nested within 50k, and each bank has its own global mean. The calibration set contains 64 images, two per main class, disjoint from the final evaluation set. Results average eight target positions and two noise draws within image. Paired pointwise 95% intervals use 2,000 hierarchical bootstrap draws over classes and then two images/class. Positive differences mean greater excess MSE with 50k.

[Paired denoising differences](sample_size_loss_comparison.csv) · [Context overlap, excluding the shared nine-token core](sample_size_context_comparison.csv).

This is a sample-size sensitivity check, not proof of convergence or independence between the two banks. Both outcomes are retained; the main 50k analysis was selected before this check, and no automatic larger-bank expansion or performance-based selection occurs.
''')


def gallery(root, fields=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    from matplotlib.patches import Rectangle
    d=read(root/'design.json');old=Path(d['pool']['gallery_parent'])/'score_comparison'
    g=read(old/'design.json');spec=g['gallery']['imagenet256'];r=read(old/'imagenet256_jit_l.json');assert digest(r['path'])==r['sha256']
    with np.load(r['path']) as z:noisy=z['noisy']
    f=get_fields(root,50000) if fields is None else fields;out=root/'score_comparison/figures';out.mkdir(parents=True,exist_ok=True);images=[]
    plt.rcParams.update({'font.size':12,'axes.titlesize':12,'pdf.fonttype':42})
    for ii,rec in enumerate(spec['records']):
        for q in rec['gallery_queries']:
            stem=f'imagenet256_pair{rec["gallery_pair"]}_image{ii%2}_q{q:03d}'
            with np.load(old/'figures'/f'{stem}_raw.npz') as z:original=z['scores'];dist=z['distance'];times=z['times']
            pooled=np.stack([f['score'][list(f['times']).index(t),q] for t in times])
            scores=np.concatenate([original[:,:2],pooled[:,None],original[:,2:]],axis=1)
            assert np.array_equal(scores[:,[0,1,3,4,5,6]],original)
            fig,axes=plt.subplots(3,9,figsize=(23,8.4),layout='constrained');rr,cc=divmod(q,16)
            titles=['Noisy input','Distance','Sensitivity\nJiT-L','Wiener\n(class-specific)','Wiener\n(pooled 50k)','JiT-L attention\n(average)','JiT-L attention','JiT-B attention\n(average)','JiT-B attention']
            for ti,t in enumerate(times):
                arr=noisy[ii*3+ti];axes[ti,0].imshow(np.clip(arr.transpose(1,2,0)/2+.5,0,1))
                axes[ti,0].add_patch(Rectangle((cc*16-.5,rr*16-.5),16,16,fill=False,ec='#ef3038',lw=2))
                dim=axes[ti,1].imshow(dist,cmap='viridis_r',norm=Normalize(0,15))
                for col,values in enumerate(scores[ti],2):
                    heat=axes[ti,col].imshow((values*256/values.sum()).reshape(16,16),cmap='magma',norm=LogNorm(.03,30),interpolation='nearest')
                for col in range(9):
                    axes[ti,col].set_xticks([]);axes[ti,col].set_yticks([])
                    if ti==0:axes[ti,col].set_title(titles[col])
                    if col:axes[ti,col].add_patch(Rectangle((cc-.5,rr-.5),1,1,fill=False,ec='#24dfd0',lw=1.5))
                axes[ti,0].set_ylabel(f't={t:g}')
            fig.colorbar(dim,ax=axes[:,1].tolist(),location='bottom',shrink=.8,pad=.025,label='Token distance')
            fig.colorbar(heat,ax=axes[:,2:].ravel().tolist(),location='bottom',shrink=.75,pad=.025,label='Score / spatial mean (display only)',extend='both')
            fig.suptitle(f'{rec["class_name"]} · {rec["image_id"]} · target token {q}',fontsize=16)
            for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=150,bbox_inches='tight',pad_inches=.04)
            plt.close(fig);save_npz(out/f'{stem}_raw.npz',scores=scores,times=times,query=q,distance=dist)
            images.append(dict(pair=rec['gallery_pair'],query=q,image=ii%2,image_id=rec['image_id'],path=out/f'{stem}.png'))
    return sorted(images,key=lambda x:(x['pair'],x['query'],x['image']))


def rebase(text,source,destination):
    def replace(m):
        target=m.group(1)
        if target.startswith(('#','http:','https:','/')):return m.group(0)
        return ']('+os.path.relpath(source/target,destination)+')'
    return re.sub(r'\]\(([^)]+)\)',replace,text)


def report(root):
    d=read(root/'design.json')
    subprocess.run([sys.executable,'-m','experiments.attention_context_transfer.report_donor_full','--report',str(root)],check=True)
    stability(root);images=gallery(root)
    def section(base):
        def link(p):return os.path.relpath(p,base)
        text='## ImageNet256: JiT-B and JiT-L\n\nPooled Wiener uses 50,000 eligible training images, 50 from each of 1,000 classes. Class-specific Wiener retains the completed full-eligible-class estimates. All seven previous loss arrays are unchanged; pooled Wiener is an additional eighth selector.\n\n### Score maps\n\nThe new pooled column shares the same field across images/classes at a fixed target and noise. Existing score columns and noisy images are preserved.\n\n'
        prior=None
        for im in images:
            key=im['pair'],im['query']
            if key!=prior:text+=f'#### Image pair {key[0]+1}, patch {key[1]}\n\n';prior=key
            text+=f'**{im["image_id"]}**\n\n![ImageNet score maps]({link(im["path"])})\n\n'
        text+='### Denoising curves\n\n'+f'![ImageNet eight-method curves]({link(root/"preview_12_48_primary.png")})\n\n'
        text+=' · '.join(f'[{label}]({link(root/file)})' for label,file in [('PDF','preview_12_48_primary.pdf'),('All times','preview_12_48_all_times.png'),('Full context range','full_context_primary.png'),('All curves and intervals','curves.csv'),('Paired comparisons','paired_contrasts.csv'),('25k/50k stability','STABILITY.md')])+'\n\n'
        return text
    canonical=Path(d['pool']['canonical_findings']);prior=canonical.read_text()
    start=prior.index('## ImageNet256:');stop=prior.index('## CelebA128:',start)
    new=prior[:start]+section(canonical.parent)+prior[stop:]
    notice='**Pooled ImageNet Wiener added:** 50k class-balanced training images, with a separate nested 25k calibration check. The previous class-specific Wiener comparison remains visible.\n\n'
    new=new.replace('# Attention experiment: current findings\n\n','# Attention experiment: current findings\n\n'+notice,1)
    (root/'FINDINGS.md').write_text(rebase(new,canonical.parent,root))
    # Preserve a review copy rather than clobber a concurrently edited findings page.
    published=False
    if digest(canonical)==d['pool']['canonical_at_submission_sha256']:
        backup=canonical.with_name('FINDINGS_before_pooled_wiener.md')
        if not backup.exists():backup.write_text(prior)
        tmp=canonical.with_suffix('.tmp');tmp.write_text(new);tmp.replace(canonical);published=True
    record(root,'PUBLICATION_COMPLETE.json',canonical_updated=published,findings_sha256=digest(root/'FINDINGS.md'),
        original_methods_preserved=True,gallery_panels=12,stability_samples=[25000,50000],
        note='If canonical_updated is false, a concurrent edit was preserved; the complete new report is in this directory.')
