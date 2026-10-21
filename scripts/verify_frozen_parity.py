"""Small recorded DEV pools; compare intermediates, scores and rankings, no GT."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from fvsr.model import resolve_manifest,load_geometry
from fvsr.controls import load_control,load_residual_geometry
from fvsr.core.replay_bound import parent_forward,top,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--native-w3',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'frozen_reference/T001'))
    import replay_bound as original
    from reference_control import ReferenceMatchedMLP
    from r4_model_bound import QueryRMSScaler
    spec=importlib.util.spec_from_file_location('original_residual',root/'frozen_reference/controls/residual_bound.py');res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res);original.ResidualParent=res.ResidualParent
    manifest=resolve_manifest(Path(a.artifact_dir)/'MANIFEST.json',a.artifact_dir)
    if manifest['role']!='ALREADY_USED_DEV_FROZEN_PARITY_ONLY':raise PermissionError('Only recorded old DEV golden inputs')
    checks=[];native_checks=[]
    for case in manifest['cases']:
        b=(load_residual_geometry if case['kind']=='list_residual' else load_geometry)(case['geometry'])
        if a.native_w3 and case['task']=='T001':
            from fvsr.backends import IVFPQAdapter
            from fvsr.core.replay_bound import cached_mask
            backend=IVFPQAdapter(b,case['native_index'],256)
            with np.load(verify(case['native_attributes']),allow_pickle=False) as z:attrs={k:z[k] for k in z.files}
            attrs['_neutral_order']=np.argsort(attrs['neutral'],kind='stable');attrs['_projection_sorted']=attrs['projection'][attrs['projection_order']]
            for item in case['packs']:
                with np.load(verify(item['input']),allow_pickle=False) as z:
                    q=z['query'];mask=cached_mask(attrs,case['native_predicate'],q);ids,ss=backend.search(q,mask,400)
                    native_checks.append(dict(qid=item['qid'],ids_exact=np.array_equal(ids,z['ids']),scores_allclose=bool(np.allclose(ss,z['native_scores'],rtol=1e-5,atol=1e-5))))
        for seed in case['seeds']:
            models=case['models'];acp=torch.load(verify(models[f'{seed}_A']),map_location='cpu',weights_only=False)
            for method in case['methods']:
                actual=load_control(b,models,seed,method)
                if method in ('FINAL','BASE_MLP'):
                    ref=original.Deployment(b,models[f'{seed}_A'],models[f'{seed}_{method}'],seed,method)
                else:
                    ref=original.Deployment.__new__(original.Deployment);ref.b=b;ref.arm='FINAL';ref.parent=original.parent_model(b,seed).eval();ref.parent.load_state_dict(acp['model'],strict=True)
                    if method!='SCC_ONLY':
                        cp=torch.load(verify(models[f'{seed}_{method}']),map_location='cpu',weights_only=False);s=cp['model']
                        ref.model=ReferenceMatchedMLP(len(b.C),8,QueryRMSScaler(s['scaler.mean'],s['scaler.std']),reference=method,fit_mean=s.get('fit_u_mean'),fit_std=s.get('fit_u_std'));ref.model.load_state_dict(s,strict=True);ref.model.eval();ref.gamma=cp['selected_gamma']
                for item in case['packs']:
                    with np.load(verify(item['input']),allow_pickle=False) as z:
                        q=z['query']@b.rotation.T;ids=z['ids'];native=z['native_scores'];key=f'{method}_{seed}'
                        if key not in z.files:key=f'Base_MLP_MATCHED_{seed}'
                        saved=z[key].copy()
                    with torch.inference_mode():
                        ap,au,az=parent_forward(actual.parent,b,q[None],ids[None],method!='BASE_MLP');rp,ru,rz=original.parent_forward(ref.parent,b,q[None],ids[None],method!='BASE_MLP')
                        stages=dict(U=torch.equal(au,ru),parent=torch.equal(ap,rp),codes=torch.equal(az,rz))
                        if method!='SCC_ONLY':
                            if method=='BASE_MLP':ap=rp=torch.from_numpy(native[None])
                            rms=torch.from_numpy(q.reshape(len(b.C),-1))[None].square().mean(-1).sqrt();valid=torch.ones_like(ap,dtype=torch.bool)
                            ax=actual.model.prepare(az,au,rms,ap,valid);rx=ref.model.prepare(rz,ru,rms,rp,valid)
                            stages.update(U_normalized=torch.equal(ax.u_scaled,rx.u_scaled),query_scaler=torch.equal(ax.query_rms_scaled,rx.query_rms_scaled),head=torch.equal(actual.model.raw_residual(ax),ref.model.raw_residual(rx)))
                    scores=actual.score_transformed(q,ids,native);expected=rp[0].numpy() if method=='SCC_ONLY' else ref.score_transformed(q,ids,native)
                    stages.update(scores=np.array_equal(scores,expected),top100=np.array_equal(top(ids,scores,100),original.top(ids,expected,100)))
                    checks.append(dict(task=case['task'],kind=case['kind'],qid=item['qid'],seed=seed,method=method,gamma=getattr(actual,'gamma',None),reference_stages=stages,
                                       historical_scores_allclose=bool(np.allclose(scores,saved,rtol=1e-5,atol=1e-5)),historical_max_abs=float(np.max(np.abs(scores-saved))),
                                       historical_top100_identical=bool(np.array_equal(top(ids,scores,100),top(ids,saved,100))),historical_top100_set_identical=set(top(ids,scores,100))==set(top(ids,saved,100))))
    source_ok=all(all(x['reference_stages'].values()) for x in checks);rank_ok=all(x['historical_top100_identical'] for x in checks);values_ok=all(x['historical_scores_allclose'] for x in checks)
    native_ok=all(x['ids_exact'] and x['scores_allclose'] for x in native_checks)
    result=dict(status='PASS' if source_ok and rank_ok and values_ok and native_ok else 'DIFFERENCE_RECORDED',same_platform_original_reference_exact=source_ok,historical_ranks_exact=rank_ok,historical_values_allclose=values_ok,checks=checks,native_checks=native_checks,torch=torch.__version__,numpy=np.__version__,new_scientific_queries=0,new_training=0,GT_opened=False,raw_database_opened=False,native_search_performed=a.native_w3)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));print('checks',len(checks))
    if not(source_ok and rank_ok and values_ok and native_ok):raise SystemExit(1)
if __name__=='__main__':main()
