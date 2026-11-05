"""CPU replay of actual S01 image-level predictions; no model execution."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from roa_v7.core import PredictionTable, calibrate
from yoto_v7.costs import registry

ATOL = 2e-12
RTOL = 1e-9

def replay(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data['rows']
    def col(key):
        return np.array([r[key] for r in rows])
    table = PredictionTable(col('image_id'), col('group_id'), col('partition'),
        col('block_id'), col('native_a'), col('native_t'),
        {b: np.array([r['candidate_prefix'].get(str(b), np.nan) for r in rows]) for b in range(1,20)})
    result = calibrate(table, registry(8))
    errors = []
    def compare(expected, actual, key='root'):
        if isinstance(expected, dict):
            assert expected.keys() == actual.keys(), key
            for k in expected: compare(expected[k], actual[k], key+'.'+k)
        elif isinstance(expected, list):
            assert len(expected)==len(actual), key
            for i,(a,b) in enumerate(zip(expected,actual)): compare(a,b,key+str(i))
        elif isinstance(expected,float):
            assert math.isfinite(actual) and math.isclose(expected,actual,abs_tol=ATOL,rel_tol=RTOL), (key,expected,actual)
            errors.append(abs(expected-actual))
        else:
            assert expected==actual,(key,expected,actual)
    compare(data['golden'],result)
    return {'CALIBRATION_REPLAY':'PASS','atol':ATOL,'rtol':RTOL,
            'maximum_absolute_error':max(errors,default=0),'result':result}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'examples/calibration_replay/S01.json')
    parser.add_argument('--output',type=Path,default=Path('replay_result.json'))
    args=parser.parse_args()
    result=replay(args.data)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='result'}))
    print('B_safe:',result['result']['B_safe'],'Selected_B:',result['result']['B_V7'])
