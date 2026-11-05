
"""Compare independently acquired real inference outputs to a real golden file."""
import argparse,json,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();x=json.loads(a.actual.read_text());g=json.loads(a.golden.read_text())
    if a.actual.resolve()==a.golden.resolve():raise ValueError('Independent execution files required')
    for k in ['checkpoint_sha256','image_sha256','image_id','native_seed','precision']:
        assert x[k]==g[k],k
    errors=[]
    for m in ['Native20','ROA']:
        u,v=x['methods'][m],g['methods'][m]
        for k in ['coordinates','model_calls','crop_forwards','batch_input_shapes']:assert u[k]==v[k],(m,k)
        assert len(u['scores'])==len(v['scores'])
        for b,c in zip(u['scores']+[u['prediction']],v['scores']+[v['prediction']]):
            assert math.isfinite(b) and math.isfinite(c) and abs(b-c)<=1e-5,(m,b,c)
            errors.append(abs(b-c))
    report={'parity':'PASS','atol':1e-5,'maximum_absolute_error':max(errors),
        'actual_environment':x.get('environment'),'golden_environment':g.get('environment')}
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))
if __name__=='__main__':main()
