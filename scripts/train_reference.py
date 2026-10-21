"""Default dry-run; explicit execution uses archived numerical training code."""
import argparse,json
from contextlib import nullcontext
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--artifact-dir');p.add_argument('--output');p.add_argument('--phase',choices=['A','B','QC','NN_RECON','NN_QUERY_MATCHED'],default='QC');p.add_argument('--arm',default='FINAL');p.add_argument('--seed',type=int,help='Caller-supplied random-state identity; no historical default is provided');p.add_argument('--reference',choices=['POOL_REF','FIT_REF']);p.add_argument('--resume',action='store_true',help='Use the hash-bound resume checkpoint in the manifest; fresh output directory');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.reference and (a.phase not in ('B','QC') or a.arm!='FINAL'):p.error('Paired reference requires phase B/QC and arm FINAL')
    recipes=json.loads((Path(__file__).resolve().parents[1]/'configs/training/original_recipes.json').read_text())
    report=dict(status='DRY_RUN_NO_OPTIMIZER',phase=a.phase,arm=a.arm,seed=a.seed,recipe=recipes['recipes'][a.phase],requirements=['Exact bound codec, database, disjoint FIT/SELECT_CAL/SELECT_CHECK cache', 'No DEV, confirmation, TEST, or unopened FRESH feedback', 'QC requires original selected B checkpoint and bound parent', 'Dependencies: install the training extra (SciPy)', 'Fresh output directory; all epoch recovery checkpoints retained'],qualification='Training equations and selection flow are included. Software checks used artificial inputs; real-data retraining was not run for this source snapshot.')
    report.update(reference=a.reference,resume=a.resume)
    if not a.execute:print(json.dumps(report,indent=2));return
    if a.seed is None:p.error('--execute requires an explicit --seed')
    if not all([a.manifest,a.artifact_dir,a.output]):p.error('Explicit execution requires manifest, artifact-dir and a fresh output directory')
    import numpy as np,torch
    from fvsr.model import resolve_manifest,load_geometry
    from fvsr.controls import load_residual_geometry
    from fvsr.core.replay_bound import verify,state_hash
    from fvsr.core import training_io
    from fvsr.core.trainer_v2 import SharedTrainer,parent_model
    m=resolve_manifest(a.manifest,a.artifact_dir)
    if m['evidence_role']!='FIT_SELECT_ONLY' or m['role_schema']!='FIT0_SELECT_CAL1_SELECT_CHECK2':raise PermissionError('Explicit FIT/SELECT-only cache required')
    with np.load(verify(m['cache']),allow_pickle=False) as z:d={k:z[k] for k in z.files}
    d.update(role_schema=m['role_schema'],dataset=m['dataset'],identity=m['cache']['sha256'],backend_identity=m['backend_identity'])
    b=(load_residual_geometry if m.get('residual') else load_geometry)(m['geometry']);db=np.load(verify(m['database']),mmap_mode='r',allow_pickle=False)
    if db.shape!=(len(b.codes),b.rotation.shape[0]):raise ValueError('Database/codec geometry mismatch')
    parent=None
    if a.phase in ('B','QC'):
        parent=parent_model(b,a.seed);cp=torch.load(verify(m['parent']),map_location='cpu',weights_only=False)
        if cp['phase']!='A' or cp['selected_state_sha']!=state_hash(cp['model']):raise ValueError('Selected parent identity')
        parent.load_state_dict(cp['model'],strict=True)
    if a.phase in ('QC','NN_QUERY_MATCHED') and 'initial' not in m:raise ValueError('Selected trained initial state required')
    context=nullcontext()
    if a.reference:
        from fvsr.reference_training import reference_factory
        if 'full_scaler_checkpoint' not in m:raise ValueError('Paired reference requires original Full scaler checkpoint')
        full=torch.load(verify(m['full_scaler_checkpoint']),map_location='cpu',weights_only=False)
        if full['phase']!='QC' or full['arm']!='FINAL' or full['selected_state_sha']!=state_hash(full['model']):raise ValueError('Full scaler checkpoint identity')
        context=reference_factory(parent,b,d,full,a.reference)
    if a.resume and 'resume' not in m:raise ValueError('Hash-bound recovery checkpoint required')
    training_io.configure(a.output)
    with context:
        trainer=SharedTrainer(b,d,db,'FORMAL',a.seed,m['cache']['sha256'],device='cpu');result=trainer.run(a.phase,Path(a.output)/'stage',A=parent,arm=a.arm,initial=m.get('initial'),resume=m.get('resume') if a.resume else None)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
