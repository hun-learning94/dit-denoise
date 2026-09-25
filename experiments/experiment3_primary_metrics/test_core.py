import unittest,tempfile,json,csv
from pathlib import Path
import numpy as np
from .core import *
from .pipeline import approved,preview,image_key
from .ratings import summarize as rating_summary

class MetricsTests(unittest.TestCase):
 def data(self):
  s={c:np.full(8,.2+.6*int(c,2)/7) for c in CODES};d={c:np.arange(8,dtype=float)+int(c,2) for c in CODES};e={c:np.full(8,1-int(c,2)/7) for c in CODES};return s,d,e
 def test_anchors_and_contrasts(self):
  s,d,e=self.data();r=bootstrap(s,d,e,repeats=40)
  self.assertAlmostEqual(r['metrics']['T']['000'],0);self.assertAlmostEqual(r['metrics']['T']['111'],1)
  self.assertAlmostEqual(r['metrics']['P']['000'],0);self.assertAlmostEqual(r['metrics']['P']['111'],1)
  self.assertEqual(r['intervals']['contrasts']['D']['QK_restoration']['ci95'],[4.,4.])
 def test_denominator_guards_and_unclipped(self):
  s,d,e=self.data();s['111']=s['000']+.001;e['000'][:]=.001;r=summarize(s,d,e)
  self.assertIsNone(r['metrics']['T']['100']);self.assertIsNone(r['metrics']['P']['100'])
  s,d,e=self.data();e['100'][:]=2;s['100'][:]=1;r=summarize(s,d,e)
  self.assertEqual(r['metrics']['P']['100'],-1);self.assertGreater(r['metrics']['T']['100'],1)
 def test_real_knn_self_exclusion(self):
  real=np.array([[1,0],[0,1],[-1,0]])
  np.testing.assert_allclose(real_radii(real,k=1),[2,2,2]);np.testing.assert_allclose(density_per_image([[1,0]],real,k=1),[2])
  with self.assertRaises(ValueError):real_radii(real,3)
 def test_probe_real_discrimination(self):
  x=np.array([[-1,.1],[-1,-.1],[1,.1],[1,-.1]]);y=np.array([0,0,1,1]);w=fit_probe(x,y,.01);p=probe_scores(x,w)
  self.assertEqual(discrimination(y,p)['auroc'],1);self.assertLess(discrimination(y,p)['log_loss'],.2)
  self.assertEqual(discrimination(y,np.full(4,.5))['auroc'],.5)
 def test_routing_identical_and_disjoint(self):
  a=np.array([[1.,0]]);b=np.array([[0.,1]])
  self.assertAlmostEqual(routing(a,a,1)['js'][0],0);self.assertEqual(routing(a,a,1)['overlap'][0],1)
  self.assertAlmostEqual(routing(a,b,1)['js'][0],np.log(2));self.assertEqual(routing(a,b,1)['overlap'][0],0)
 def test_approval_and_deferred_grid(self):
  with self.assertRaises(RuntimeError):approved({'status':'awaiting_user_review'})
  r=preview([dict(epoch=50,budget_fid_complete=True,selection_paired_epochs=[25,50,75])]);self.assertEqual(len(r['deferred']),1)
 def test_source_cache_reuse_and_invalidation(self):
  c=dict(images=512,latent_seed=1,sampler_steps=50,precision='fp32',generation_batch=16)
  self.assertEqual(image_key(c,{'sha256':'x','path':'a'}),image_key(c,{'sha256':'x','path':'b'}))
  self.assertNotEqual(image_key(c,{'sha256':'x'}),image_key(dict(c,sampler_steps=51),{'sha256':'x'}))
 def test_rating_denominator_and_missing(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);key=[];ratings=[]
   for c in CODES:
    for i in range(2):
     ident=c+str(i);key.append(dict(item_id=ident,cube='x',code=c,latent=i))
     ratings.append(dict(item_id=ident,coherent='yes' if i==0 else 'uncertain',domain='target',rater='adjudicator'))
   (root/'key.json').write_text(json.dumps(key))
   with (root/'ratings.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=ratings[0]);w.writeheader();w.writerows(ratings)
   r=rating_summary(root/'key.json',root/'ratings.csv',20)['x'];self.assertEqual(r['metrics']['C']['111'],.5);self.assertEqual(r['metrics']['J']['111'],.5)
   (root/'key.json').write_text(json.dumps(key[:-1]))
   with self.assertRaises(ValueError):rating_summary(root/'key.json',root/'ratings.csv',20)
if __name__=='__main__':unittest.main()
