"""Real frozen YOTO inference. Requires separately authorized model assets."""
import argparse
import json
from pathlib import Path
from yoto_v7.scoring import YotoScorer, ImageSpec, image_mean, sha256
from yoto_v7.sampling import native_coordinates, sobol_coordinates

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,default=Path('demo_result.json'))
    a=p.parse_args()
    import torch, platform
    from datetime import datetime, timezone
    torch.set_num_threads(8)
    lock=json.loads((Path(__file__).parent/'configs/BUDGET_LOCK.json').read_text())
    checkpoint_hash=lock['source_sessions'][0]['checkpoint_sha256']
    scorer=YotoScorer(dataset='KONIQ',asset_root=a.asset_root,data_root=a.image.resolve().parent,
        checkpoint=a.checkpoint,checkpoint_sha256=checkpoint_hash,device=a.device,crop_batch_size=8)
    image=scorer.load_image(ImageSpec(a.image.name,a.image.name))
    native,seed=native_coordinates(*image.shape[1:],'KONIQ','S01','FORMAL_TEST_NATIVE',a.image.name,0)
    result={'checkpoint_sha256':checkpoint_hash,'image_sha256':sha256(a.image),
            'image_id':a.image.name,'device':a.device,'native_seed':seed,'precision':'FP32','methods':{}}
    result['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'python':platform.python_version(),'utc':datetime.now(timezone.utc).isoformat()}
    result['source_sha256']={str(p.relative_to(Path(__file__).parent)):sha256(p)
        for p in [Path(__file__),*sorted((Path(__file__).parent/'yoto_v7').glob('*.py'))]}
    for method,coords in [('Native20',native),('ROA',sobol_coordinates(*image.shape[1:],lock['B_fixed']))]:
        out=scorer.score_coordinates(image,coords)
        result['methods'][method]={**out,'coordinates':coords,'prediction':image_mean(out['scores'])}
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(a.output)

if __name__=='__main__':main()
