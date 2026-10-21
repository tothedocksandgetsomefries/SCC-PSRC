"""Shared epoch trainer: exact selected-state continuation, explicit source recipes and complete recovery."""
import copy,json,os,random,time,hashlib
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .training_io import ROOT,identity,atomic,record,verify,charge,stop_requested
from .parent_bound import Parent
from .residual_bound import ResidualParent
from .r4_model_bound import MatchedMLP,QueryRMSScaler,masked_standardize,operation_smooth_l1
from .training_reference import NNDecoder,aq_fixed,RECIPES,snapshot,restore,checkpoint,step
from .training_primitives import sample_positive_ids,static_random_negative_ids,zscore_tensor,stress_weights,operation_weights,select_epoch
from .selection import GAMMAS,select_gamma
from .replay_bound import top
from .child_models_v2 import CHILD_ARMS,child_model
def model_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def state_hash(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):
        a=v.detach().cpu().contiguous().detach().cpu().numpy();h.update(identity(dict(name=k,shape=list(a.shape),dtype=str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()
class EpochSelector:
    def __init__(self,phase,initial):
        self.phase=phase;self.initial=initial;self.best=initial;self.epoch=0;self.stale=0
    def update(self,epoch,metrics):
        if self.phase in ('QC','NN_QUERY_MATCHED'):
            improved,safe,J=select_epoch(metrics,self.initial,dict(J=.5*(self.best['R100']+self.best['strictR10'])))
        elif self.phase=='NN_RECON':safe=True;improved=metrics['loss']<self.best['loss']
        else:safe=True;improved=metrics['R100']>self.best['R100']
        if improved:self.best=metrics;self.epoch=epoch;self.stale=0
        else:self.stale+=1
        return bool(improved),bool(safe)
    def stop(self,epoch,recipe):return epoch>=recipe['min_epochs'] and self.stale>=recipe['patience']
def parent_model(b,seed):
    return ResidualParent(b.C,b.coarse,seed) if b.residual else Parent('A_GLOBAL',b.C,seed)
def parent_forward(A,b,q,ids,calibrated=True):
    device=next(A.parameters()).device
    z=torch.as_tensor(b.codes[np.asarray(ids)].astype('int64'),device=device);qt=torch.as_tensor(q,dtype=torch.float32,device=device)
    if b.residual:
        lists=torch.as_tensor(b.lists[np.asarray(ids)].astype('int64'),device=device);p,u=A.parts(qt,lists,z,calibrated)
    elif calibrated:p,u,_,_=A.score(qt,torch.zeros((len(q),1),dtype=torch.long,device=device),z)
    else:
        lut=A.base_lut(qt);u=torch.stack([lut[:,m].gather(1,z[:,:,m]) for m in range(A.m)],-1)
        p=torch.zeros(u.shape[:2],device=device); 
        for m in range(A.m):p=p+u[:,:,m]
    return p,u,z
def feature_cache(A,b,d,calibrated):
    device=next(A.parameters()).device
    ps=[];us=[]
    with torch.no_grad():
        for off in range(0,len(d['qids']),16):
            ii=d['ids'][off:off+16];safe=np.where(d['valid'][off:off+16],ii,0)
            p,u,z=parent_forward(A,b,d['query'][off:off+16],safe,calibrated)
            if not calibrated:p=torch.as_tensor(d['base'][off:off+16],device=device).clone()
            v=torch.as_tensor(d['valid'][off:off+16],device=device);p[~v]=0;u[~v]=0;ps.append(p);us.append(u)
    p=torch.cat(ps);u=torch.cat(us);v=torch.as_tensor(d['valid'],device=device)
    codes=b.codes[np.where(d['valid'],d['ids'],0)].astype('int64')
    rms=torch.as_tensor(d['query'].reshape(len(d['qids']),len(b.C),-1),device=device).square().mean(-1).sqrt()
    target=masked_standardize(torch.as_tensor(d['exact'],device=device),v)[0]-masked_standardize(p,v)[0]
    return dict(inputs=(torch.as_tensor(codes,device=device),u,rms,p,v),target=target,parent_state_sha=state_hash(A.state_dict()))
def metric_summary(d,rows,scores):
    values=[]
    for i,s in zip(rows,scores):
        rank=top(d['ids'][i],s,100,d['valid'][i]);g=d['gt'][i];g=g[g>=0];legal=int(d['legal_count'][i])
        rec={}
        for k,name in [(100,'R100'),(10,'strictR10')]:
            den=min(k,legal);rec[name]=len(set(g[:den])&set(rank[:k]))/den if den else None
        values.append(dict(qid=int(d['qids'][i]),predicate=str(d['predicate'][i]),**rec))
    means={}
    for name in ['R100','strictR10']:
        qmeans=[]
        for q in sorted({x['qid'] for x in values}):
            pmeans=[]
            for p in sorted({x['predicate'] for x in values if x['qid']==q}):
                a=[x[name] for x in values if x['qid']==q and x['predicate']==p and x[name] is not None]
                if a:pmeans.append(np.mean(a))
            if pmeans:qmeans.append(np.mean(pmeans))
        if not qmeans:raise ValueError('No estimable SELECT qid')
        means[name]=float(np.mean(qmeans))
    return means
def initial_gamma(metrics,parent):
    eligible=[g for g in GAMMAS if metrics[g]['R100']-parent['R100']>=-.001 and metrics[g]['strictR10']-parent['strictR10']>=-.0025]
    if not eligible:raise ValueError('Gamma0 parent fallback missing')
    return sorted(eligible,key=lambda g:(-metrics[g]['R100'],g))[0]
class SharedTrainer:
    def __init__(self,b,d,db,recipe_mode,seed,source_id,device='cpu',max_updates=None):
        if recipe_mode not in ('TECHNICAL_QUALIFICATION','FORMAL'):raise PermissionError('Explicit recipe role')
        if set(np.unique(d['role']))!={0,1,2}:raise ValueError('Expected local FIT/CAL/CHECK role encoding; role2 is SELECT_CHECK, not DEV')
        # Local role2 identity must explicitly name SELECT_CHECK in source schema.
        if d.get('role_schema')!='FIT0_SELECT_CAL1_SELECT_CHECK2':raise PermissionError('Ambiguous role2')
        self.device=torch.device(device);self.max_updates=max_updates
        if max_updates is not None and (recipe_mode!='TECHNICAL_QUALIFICATION' or not 1<=max_updates<=300):raise ValueError('Bounded technical updates only')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        self.b=b;self.d=d;self.db=db;self.mode=recipe_mode;self.seed=seed;self.source_id=source_id
        self.fit=np.flatnonzero(d['role']==0);self.select=np.flatnonzero(d['role']!=0)
        if set(d['qids'][self.fit])&set(d['qids'][self.select]):raise PermissionError('Role overlap')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def recipe(self,phase):
        r=copy.deepcopy(RECIPES[phase]);r['phase']=phase;r['mode']=self.mode
        if phase=='NN_RECON':r.update(min_epochs=1,patience=300)
        if self.mode=='TECHNICAL_QUALIFICATION':r.update(max_epochs=2,min_epochs=2,patience=2)
        return r
    def run(self,phase,out,A=None,arm='FINAL',initial=None,resume=None,pause_after_epoch=None):
        if arm not in CHILD_ARMS:raise NotImplementedError('Unimplemented method arm must not silently become a different scorer: '+str(arm))
        out=Path(out);out.mkdir(parents=True,exist_ok=False);d=self.d;b=self.b;seed=self.seed
        device=self.device
        if A is not None:A=A.to(device)
        r=self.recipe(phase);rng_seed=seed+20260910 if phase in ('QC','NN_QUERY_MATCHED') else seed
        torch.manual_seed(rng_seed);np.random.seed(rng_seed);random.seed(rng_seed);rng=np.random.default_rng(seed+1000 if phase=='A' else rng_seed)
        cache=None
        if phase=='A':model=parent_model(b,seed)
        elif phase.startswith('NN'):
            model=NNDecoder(b.C)
            if initial:model.load_state_dict(torch.load(verify(initial),map_location='cpu',weights_only=False)['model'],strict=True)
        else:
            if A is None:raise ValueError('Explicit bound parent required')
            cache=feature_cache(A,b,d,arm!='BASE_MLP')
            scaler=QueryRMSScaler.fit(cache['inputs'][2][self.fit],role='FIT_TRAIN')
            model=child_model(len(b.C),8,scaler,arm)
            if initial:model.load_state_dict(torch.load(verify(initial),map_location='cpu',weights_only=False)['model'],strict=True)
            if phase=='QC' and initial is None:raise ValueError('QC cannot start random')
        model=model.to(device)
        fit=self.fit.copy();excluded=[]
        if phase=='A':
            eligible=(d['labels100'][fit]&d['valid'][fit]).any(1)&((~d['labels100'][fit])&d['valid'][fit]).any(1)
            excluded=fit[~eligible].tolist()
            if excluded and d['dataset']!='W3':raise ValueError('Non-W3 A ineligible operation')
            fit=fit[eligible]
            ww,sep,scale=stress_weights(d['exact'][fit],d['labels100'][fit],d['valid'][fit])
            weight={int(i):float(w) for i,w in zip(fit,ww)}
        else:
            ww=operation_weights(d,fit);weight={int(i):float(w) for i,w in zip(fit,ww)}
        gamma=1.
        if phase=='QC':
            gamma=float(torch.load(verify(initial),map_location='cpu',weights_only=False)['selected_gamma'])
        config=dict(recipe=r,seed=seed,rng_seed=rng_seed,phase=phase,arm=arm,source_id=self.source_id,
                    device=str(device),precision='FP32_TF32_OFF_AMP_OFF',max_updates=self.max_updates,
                    data_identity=d['identity'],backend_identity=d['backend_identity'],initial=initial,
                    parent_state_sha=cache['parent_state_sha'] if cache else None,gamma=gamma,excluded_fit_operations=excluded)
        atomic(out/'CONFIG.json',config)
        def inputs(rows):return tuple(t[rows] for t in cache['inputs'])
        def scores(rows,g=gamma):
            model.eval();outsc=[]
            with torch.no_grad():
                for i in rows:
                    if phase=='A':
                        p,_,_=parent_forward(model,b,d['query'][i:i+1],np.where(d['valid'][i:i+1],d['ids'][i:i+1],0))
                        s=p[0].detach().cpu().numpy()
                    elif phase.startswith('NN'):
                        z=torch.as_tensor(b.codes[np.maximum(d['ids'][i],0)].astype('int64'),device=device);x=model.decode(z)
                        s=-(x-torch.as_tensor(d['query'][i],device=device)[None]).square().sum(-1).detach().cpu().numpy()
                    else:s=model(*inputs([i]),gamma=g)[0].detach().cpu().numpy()
                    outsc.append(s.copy())
            return np.asarray(outsc)
        learn_ids=np.asarray(d['db_learn_ids'],dtype=np.int64);cal_ids=np.asarray(d['db_cal_ids'],dtype=np.int64)
        if not len(learn_ids) or not len(cal_ids) or len(set(learn_ids))!=len(learn_ids) or len(set(cal_ids))!=len(cal_ids) or set(learn_ids)&set(cal_ids):raise ValueError('Disjoint unique bound DB_LEARN/CAL needed')
        if min(learn_ids.min(),cal_ids.min())<0 or max(learn_ids.max(),cal_ids.max())>=len(self.db):raise ValueError('DB role ID range')
        def evaluation(rows):
            eval_tick=time.perf_counter()
            m=metric_summary(d,rows,scores(rows))
            with torch.no_grad():
                if phase=='NN_RECON':
                    z=torch.as_tensor(b.codes[cal_ids].astype('int64'),device=device);x=torch.as_tensor(np.ascontiguousarray(self.db[cal_ids]@b.rotation.T),device=device)
                    m['loss']=float(F.mse_loss(model.decode(z),x))
                elif phase.startswith('NN'):
                    v=torch.as_tensor(d['valid'][rows],device=device);pr=masked_standardize(torch.as_tensor(scores(rows),device=device),v)[0];tar=masked_standardize(torch.as_tensor(d['exact'][rows],device=device),v)[0]
                    m['loss']=float((operation_smooth_l1(pr,tar,v)*torch.as_tensor(operation_weights(d,rows),device=device)).mean())
                elif phase=='A':m['loss']=None
                else:
                    losses=operation_smooth_l1(model.residual(*inputs(rows)),cache['target'][rows],cache['inputs'][-1][rows])
                    m['loss']=float((losses*torch.as_tensor(operation_weights(d,rows),device=device)).mean())
            if device.type=='cuda':torch.cuda.synchronize(device)
            m['evaluation_seconds']=time.perf_counter()-eval_tick
            return m
        initial_metrics=evaluation(self.select);selector=EpochSelector(phase,initial_metrics);best=model_state(model)
        history=[dict(epoch=0,SELECT=initial_metrics,FIT=evaluation(fit),selected=True,updates=0)];updates=0;start=1
        charge('OPTIMIZER_START',str(out),details=dict(config=record(out/'CONFIG.json'),role=self.mode,phase=phase,initial_state_sha=state_hash(model.state_dict())))
        cls=getattr(torch.optim,r['optimizer']);opt=cls(model.parameters(),lr=r['lr'],weight_decay=r['weight_decay'])
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=r['plateau_patience']) if phase=='NN_RECON' else None
        if resume:
            saved=torch.load(verify(resume),map_location='cpu',weights_only=False)
            restore(saved,model,opt,sched,config);rng.bit_generator.state=saved['sampler']['generator']
            selector=EpochSelector(phase,saved['selected']['initial']);selector.__dict__.update(saved['selected']['selector'])
            initial_metrics=saved['selected']['initial'];best=saved['selected']['best_model'];history=saved['selected']['history'];updates=saved['selected']['updates'];start=saved['next_epoch']
        def save_recovery(epoch):
            selected=dict(initial=initial_metrics,selector=copy.deepcopy(selector.__dict__),best_model=best,history=history,updates=updates)
            cp=snapshot(model,opt,sched,epoch,dict(generator=rng.bit_generator.state,cursor=0),selected,config)
            saved=checkpoint(out/f'epoch{epoch:03d}.pt',cp)
            from .training_storage import retain_recovery
            retain_recovery(out,keep_latest=2)
            if os.environ.get('PE3_SERVER_RECOVERY_CHECK')=='1' and epoch==1:
                from .training_checks import verify_recovery
                verify_recovery(saved,model,opt,sched,config,rng,out)
            return saved
        latest=save_recovery(start-1)
        for epoch in range(start,r['max_epochs']+1):
            model.train();losslogs=[];tick=time.perf_counter()
            order=rng.permutation(fit) if phase=='A' else np.random.default_rng(rng_seed+epoch).permutation(fit)
            if phase=='NN_RECON':order=np.random.default_rng(rng_seed+epoch).permutation(learn_ids)
            for off in range(0,len(order),r['batch']):
                if self.max_updates is not None and updates>=self.max_updates:break
                if device.type=='cuda':torch.cuda.synchronize(device)
                step_tick=time.perf_counter()
                rows=order[off:off+r['batch']]
                if phase=='A':
                    chosen=[]
                    for i in rows:
                        p,_=sample_positive_ids('CANDIDATE_ONLY',d['ids'][i],d['labels100'][i],d['gt'][i],rng)
                        n=static_random_negative_ids(d['ids'][i],d['valid'][i],d['labels100'][i],d['base'][i],rng)
                        if len(p)!=64 or len(n)!=64:raise ValueError('64x64 sample contract')
                        chosen.append(np.r_[p,n])
                    chosen=np.asarray(chosen);q=d['query'][rows];s,_,_=parent_forward(model,b,q,chosen)
                    native,_,_=parent_forward(model,b,q,chosen,False)
                    pair=F.softplus(1-s[:,:64,None]+s[:,None,64:])
                    per=(pair*torch.ones((len(rows),1,64),device=device)).sum((1,2))/64./64. if d['dataset']=='W3' else pair.mean((1,2))
                    loss=(per*torch.tensor([weight[int(i)] for i in rows],device=device)).mean()+.01*(zscore_tensor(s)-zscore_tensor(native)).square().mean()
                elif phase=='NN_RECON':
                    z=torch.as_tensor(b.codes[rows].astype('int64'),device=device);x=torch.as_tensor(np.ascontiguousarray(self.db[rows]@b.rotation.T),device=device)
                    loss=F.mse_loss(model.decode(z),x)
                elif phase=='NN_QUERY_MATCHED':
                    z=torch.as_tensor(b.codes[np.where(d['valid'][rows],d['ids'][rows],0)].astype('int64'),device=device)
                    x=model.decode(z.reshape(-1,len(b.C))).reshape(len(rows),d['ids'].shape[1],-1)
                    pred=-(x-torch.as_tensor(d['query'][rows],device=device)[:,None]).square().sum(-1);v=torch.as_tensor(d['valid'][rows],device=device)
                    pred=masked_standardize(pred,v)[0];target=masked_standardize(torch.as_tensor(d['exact'][rows],device=device),v)[0]
                    loss=(operation_smooth_l1(pred,target,v)*torch.tensor([weight[int(i)] for i in rows],device=device)).mean()
                else:
                    v=cache['inputs'][-1][rows]
                    loss=(operation_smooth_l1(model.residual(*inputs(rows)),cache['target'][rows],v)*torch.tensor([weight[int(i)] for i in rows],device=device)).mean()
                entry=step(model,opt,loss,r['clip'])
                if device.type=='cuda':torch.cuda.synchronize(device)
                entry['complete_step_seconds']=time.perf_counter()-step_tick
                losslogs.append(entry);updates+=1
            ev=evaluation(self.select)
            if sched:sched.step(ev['loss'])
            improved,safe=selector.update(epoch,ev)
            if improved:best=model_state(model)
            history.append(dict(epoch=epoch,SELECT=ev,FIT=evaluation(fit),selected=improved,selected_epoch=selector.epoch,
                 protection=safe,updates=updates,loss=float(np.mean([x['loss'] for x in losslogs])),
                 complete_steps_seconds=[x['complete_step_seconds'] for x in losslogs],
                 gradient_norm_max=max(x['grad_norm'] for x in losslogs),clip_fraction=float(np.mean([x['clipped'] for x in losslogs])),
                 seconds=time.perf_counter()-tick))
            latest=save_recovery(epoch);atomic(out/f'epoch{epoch:03d}.json',history[-1])
            print(json.dumps(dict(phase=phase,arm=arm,epoch=epoch,selected=selector.epoch,updates=updates)),flush=True)
            if stop_requested() or (pause_after_epoch is not None and epoch>=pause_after_epoch):
                atomic(out/'INTERRUPTED.json',dict(status='PAUSED_AT_EPOCH_BOUNDARY',recovery=latest,config=record(out/'CONFIG.json')))
                return dict(status='PAUSED_AT_EPOCH_BOUNDARY',recovery=latest)
            if selector.stop(epoch,r) or (self.max_updates is not None and updates>=self.max_updates):break
        last_hash=state_hash(model.state_dict());model.load_state_dict(best,strict=True);model.eval()
        selected_gamma=gamma
        if phase=='B':
            met={g:metric_summary(d,self.select,scores(self.select,g)) for g in GAMMAS}
            selected_gamma=initial_gamma(met,metric_summary(d,self.select,cache['inputs'][3][self.select].detach().cpu().numpy()))
        elif phase=='QC':
            met={g:metric_summary(d,self.select,scores(self.select,g)) for g in GAMMAS}
            mm={g:dict(R100=m['R100'],R10=m['strictR10']) for g,m in met.items()}
            native=metric_summary(d,self.select,d['base'][self.select])
            selected_gamma=select_gamma(mm,dict(R100=native['R100'],R10=native['strictR10']),gamma)['selected_gamma']
        payload=dict(model=model_state(model),selected_epoch=selector.epoch,selected_gamma=selected_gamma,phase=phase,arm=arm,
                     config=config,initial=initial,selected_state_sha=state_hash(model.state_dict()),source_id=self.source_id)
        selected=checkpoint(out/'SELECTED.pt',payload)
        atomic(out/'HISTORY.json',dict(role=self.mode,epochs=history,source=record(__file__)))
        result=dict(status='COMPLETE',role=self.mode,phase=phase,arm=arm,selected=selected,selected_epoch=selector.epoch,
              epochs_completed=history[-1]['epoch'],optimizer_updates=updates,last_recovery=latest,
              last_state_sha=last_hash,selected_state_sha=payload['selected_state_sha'],selected_gamma=selected_gamma,
              initial_checkpoint=initial,scientific_effect_claim=False)
        atomic(out/'RESULT.json',result);return result
