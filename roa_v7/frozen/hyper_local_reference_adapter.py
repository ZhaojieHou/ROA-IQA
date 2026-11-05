#!/usr/bin/env python3
"""Read-only, CPU-only V5 symmetric fixed-sequence replay on two FROZEN local checkpoints.

Required: ROA_CALIBRATION_MODE_LOCAL_V2_AI_REVIEW.zip (original, unmodified).
No GPU, no training, no MOS, no ROA Core edits, and no test data.
The 32 hash-block jackknife remains exploratory (not proven 95% finite-sample).

Fixed design: S01/S02, seeds 20261010/11/12, Msel=1024,
Mval=2048, lambda_proposal_safe=0, lambda_proposal_aggressive=.25,
lambda_validation=.25, Spearman AND Pearson.

Frozen V3 symmetric source results are checked *before* accepting V5 comparisons.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, t as student_t

SEEDS=(20261010,20261011,20261012)
SESSIONS=('S01','S02')
B_NATIVE=25
L_NATIVE=50
M_SELECTION=1024
M_VALIDATION=2048
VAL_LAMBDA=.25
N_HASH_BLOCKS=32
# Established in V3 SYMMETRIC_TWO_NATIVE_HALVES_36.csv; all used in prior exploration.
EXPECTED={
 ('S01',20261010): {'B':13,'S_UCB':-0.000089,'P_UCB':-0.000075},
 ('S01',20261011): {'B':15,'S_UCB':-0.000134,'P_UCB':-0.000091},
 ('S01',20261012): {'B':12,'S_UCB':-0.000065,'P_UCB':-0.000053},
 ('S02',20261010): {'B':15,'S_UCB':-0.000146,'P_UCB':-0.000139},
 ('S02',20261011): {'B':14,'S_UCB':-0.000152,'P_UCB':-0.000128},
 ('S02',20261012): {'B':12,'S_UCB':-0.000072,'P_UCB':-0.000053},
}
CHECKPOINTS={
 'S01':'A30874B86FDF7E2CB8C2F686ECB011FD902D21B0BF9C83AA894B72BA6C8B3461',
 'S02':'6DD367BCCD0A611723EFF8D6F6403BE0F8BCF9FF28D1511C38D92227029C97FA',
}

def sha(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()

def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def correlation(x,y,rank):
    if rank:
        x=rankdata(x,method='average'); y=rankdata(y,method='average')
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    x=x-x.mean();y=y-y.mean()
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    if denominator<=0:
        raise ValueError('Degenerate Spearman or Pearson variance')
    return float((x@y)/denominator)

def risk(A,T,C,lam):
    result=[]
    for rank in (True,False):  # Spearman, Pearson
        r_nat=correlation(A,T,rank)
        r_b=.5*(correlation(C,A,rank)+correlation(C,T,rank))
        result.append((1.+lam)*r_nat-r_b-lam)
    return np.asarray(result)

def proposal(A,T,C,lam):
    for b in range(1,25):
        if np.all(risk(A,T,C[:,b-1],lam)<=0):
            return b
    return B_NATIVE

def validate(A,T,C,ids,session,seed):
    blocks=np.asarray([int(sha(f'B|KONIQ|{session}|{seed}|{v}')[:10],16)%N_HASH_BLOCKS for v in ids])
    groups=np.unique(blocks)
    if len(groups)!=N_HASH_BLOCKS:
        raise ValueError('hash-block count differs from frozen V3')
    point=risk(A,T,C,VAL_LAMBDA)
    jackknife=np.asarray([risk(A[blocks!=k],T[blocks!=k],C[blocks!=k],VAL_LAMBDA) for k in groups])
    se=np.sqrt((len(groups)-1)/len(groups)*np.sum((jackknife-jackknife.mean(axis=0))**2,axis=0))
    upper=point+student_t.ppf(.975,len(groups)-1)*se
    return point,upper,bool(np.all(upper<=0))

def npz_contents(z,session):
    name=f'ROA_CALIBRATION_MODE_LOCAL_V2_AI_REVIEW/experiment/{session}_ATOMS_L50.npz'
    raw=z.read(name)
    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
        required=['image_id','group_id','native50','sobol24','checkpoint_sha256','split_sha256','native_repeat_id','native_observation_index']
        missing=[x for x in required if x not in data]
        if missing:
            raise ValueError(f'Missing keys in {name}: {missing}')
        obs=data['native50'];sobol=data['sobol24'];ids=data['image_id'];grp=data['group_id']
        ckpt=str(data['checkpoint_sha256'].item());split=str(data['split_sha256'].item())
        repeat=data['native_repeat_id'];idx=data['native_observation_index']
    if obs.shape!=(8058,50) or sobol.shape!=(8058,24) or len(set(map(str,ids)))!=8058:
        raise ValueError(f'Unexpected local source shape/identity for {session}')
    if not (list(repeat[:25])==[0]*25 and list(repeat[25:])==[1]*25 and
            list(idx[:25])==list(range(25)) and list(idx[25:])==list(range(25))):
        raise ValueError('Native half-repeat identity mismatch')
    if ckpt.upper()!=CHECKPOINTS[session]:
        raise ValueError(f'Checkpoint SHA mismatch at {session}: {ckpt}')
    A=obs[:,:25].mean(1);T=obs[:,25:].mean(1)
    C=np.cumsum(sobol,axis=1)/np.arange(1,25)
    return ids,grp,A,T,C,ckpt,split,hashlib.sha256(raw).hexdigest()

def main(source,output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:raise ValueError('Corrupted ZIP')
        for s in SESSIONS:
            ids,groups,A,T,C,checkpoint,split,member_sha=npz_contents(z,s)
            for seed in SEEDS:
                order=np.argsort([sha(f'SELECT|{s}|{seed}|{v}') for v in ids])
                select=order[:M_SELECTION]; val=order[M_SELECTION:M_SELECTION+M_VALIDATION]
                if set(map(str,ids[select]))&set(map(str,ids[val])):
                    raise ValueError('Selection-validation content overlap')
                Bs=proposal(A[select],T[select],C[select],0.0)
                Ba=proposal(A[select],T[select],C[select],.25)
                if Ba>Bs:raise AssertionError('lambda inclusion fails')
                if Bs==25:
                    spass=False;spt=np.array([math.nan,math.nan]);su=spt
                else:
                    spt,su,spass=validate(A[val],T[val],C[val,Bs-1],ids[val],s,seed)
                expected=EXPECTED[(s,seed)]
                if Bs!=expected['B'] or not spass:
                    raise RuntimeError(f'BASELINE_PARITY_FAIL {s}/{seed}: B={Bs}, pass={spass}, expected {expected}')
                # V3 only reported truncated upper values; permit rounding within 2e-6.
                if abs(su[0]-expected['S_UCB'])>2e-6 or abs(su[1]-expected['P_UCB'])>2e-6:
                    raise RuntimeError(f'BASELINE_UCB_PARITY_FAIL {s}/{seed}: actual={su}, expected={expected}')
                if Ba==Bs:
                    apt,au,apass=spt,su,spass
                elif spass and Ba<25:
                    apt,au,apass=validate(A[val],T[val],C[val,Ba-1],ids[val],s,seed)
                else:
                    apt=np.array([math.nan,math.nan]);au=apt;apass=False
                B_old=Bs if spass else B_NATIVE
                B_new=(Ba if apass else Bs) if spass else B_NATIVE
                if B_new>B_old:raise AssertionError('V5 should not choose larger B on paired data')
                rows.append(dict(session=s,seed=seed,checkpoint_sha256=checkpoint,split_sha256=split,
                   atom_npz_sha256=member_sha,B_safe=Bs,B_aggressive=Ba,
                   safe_validation_pass=spass,aggressive_validation_pass=apass,
                   B_original=B_old,B_fixed_sequence=B_new,delta_B=B_old-B_new,
                   S_UCB_safe=su[0],P_UCB_safe=su[1],S_UCB_aggressive=au[0],P_UCB_aggressive=au[1],
                   selected_content_group_count=len(set(map(str,groups[select]))),
                   validated_content_group_count=len(set(map(str,groups[val]))),
                   baseline_parity='PASS',scope='EXPLORATORY_ALREADY_EXPOSED_LOCAL_CHECKPOINTS'))
                print(f'{s}/{seed} baseline={Bs} aggressive={Ba} new={B_new}, delta={B_old-B_new}, parity=PASS',flush=True)
    target=output/'LOCAL_V5_SYMMETRIC_6_REPLAYS.csv'
    with target.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'study_status':'EXPLORATORY_NOT_PROSPECTIVE','source_zip_sha256':file_sha256(source),
      'source_zip_basename':source.name,'n':len(rows),'parity_all':all(x['baseline_parity']=='PASS' for x in rows),
      'B_old':[int(x['B_original']) for x in rows], 'B_new':[int(x['B_fixed_sequence']) for x in rows],
      'median_old':float(np.median([x['B_original'] for x in rows])),
      'median_new':float(np.median([x['B_fixed_sequence'] for x in rows])),
      'B_le_10_old':sum(x['B_original']<=10 for x in rows),
      'B_le_10_new':sum(x['B_fixed_sequence']<=10 for x in rows),
      'improved_replays':sum(x['delta_B']>0 for x in rows),
      'mean_delta_B':float(np.mean([x['delta_B'] for x in rows])),
      'source':'LOCAL_TWO_CHECKPOINTS_NATIVE50_AND_SOBOL24',
      'statistical_limit':'32 hash-block jackknife approximate; no finite-sample 95% proof',
      'inference_limit':'No new MOS or new GPU inference examined',
      'promotion_gate':'Do not promote without independent checkpoint and source-to-target performance checks'}
    (output/'LOCAL_V5_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--source-zip',type=Path,required=True)
    pa.add_argument('--out',type=Path,required=True)
    a=pa.parse_args()
    main(a.source_zip,a.out)
