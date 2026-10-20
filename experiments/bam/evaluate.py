"""Experimental BAM frame localization, fixed argmax threshold, no GT tuning."""
import sys, json, argparse, math, time, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bam_inference import BAM

class AttributionConfig:
    pass


def load(path):
    # No unrestricted pickle loading of the downloaded checkpoint.
    with torch.serialization.safe_globals([argparse.Namespace, (AttributionConfig,'utils.tool.Attribution_Config')]):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
    args=ckpt['hyper_parameters']['args'];config=ckpt['hyper_parameters']['config']
    model=BAM(args,config)
    state={k.removeprefix('model.'):v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(state,strict=True)
    del ckpt,state
    return model.eval()


def intervals(rows,key,threshold=.5):
    result=[]
    for r in rows:
        if r[key] < threshold:continue
        if result and abs(result[-1][1]-r['start'])<1e-5:result[-1][1]=r['end']
        else:result.append([r['start'],r['end']])
    return result


def predict(model,x):
    rows=[];sr=16000;hop=2560;chunk=64*hop
    # Fixed 10.24s chunks constrain quadratic graph memory; no use of answers.
    for start in range(0,len(x),chunk):
        clip=x[start:start+chunk]
        pad=int(math.ceil(len(clip)/hop)*hop)
        inp=torch.from_numpy(np.pad(clip,(0,pad-len(clip)))).float().unsqueeze(0)
        with torch.inference_mode():
            logits,boundary=model(inp)
            fake=torch.softmax(logits,dim=-1)[0,:,0].cpu().numpy()
            boundary=boundary[0].cpu().numpy()
        for i,(score,b) in enumerate(zip(fake,boundary)):
            a=(start+i*hop)/sr;end=min((start+(i+1)*hop)/sr,len(x)/sr)
            if a>=end:continue
            rows.append({'start':round(a,5),'end':round(end,5),'fake_score':float(score),'boundary_score':float(b)})
    return rows


if __name__=='__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/bam/model.ckpt')
    options=parser.parse_args()
    out=options.suite.resolve()
    model=load(options.checkpoint)
    print('Strict checkpoint loading succeeded',flush=True)
    predictions={}
    t=time.monotonic()
    for p in sorted((out/'测试音频').glob('*.wav')):
        x,sr=sf.read(p,dtype='float32');assert sr==16000
        frames=predict(model,x)
        predictions[p.name]={'frames':frames,'predicted_intervals':intervals(frames,'fake_score'),'boundary_regions':intervals(frames,'boundary_score')}
        (out/'BAM原始输出.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2))
        print(p.name,'predicted',predictions[p.name]['predicted_intervals'],'elapsed',round(time.monotonic()-t,1),flush=True)
    truth=json.loads((out/'核对答案.json').read_text())['cases']
    overlap=lambda a,b:max(0,min(a[1],b[1])-max(a[0],b[0]))
    rows=[]
    for gt in truth:
        pred=predictions[gt['file']];actual=gt['synthetic_intervals'];ps=pred['predicted_intervals']
        tp=sum(overlap(a,b) for a in actual for b in ps)
        actual_seconds=sum(b-a for a,b in actual);pred_seconds=sum(b-a for a,b in ps)
        rows.append({**gt,'predicted_intervals':ps,'overlap_hit':tp>0 if actual else None,'true_positive_seconds':tp,'false_positive_seconds':pred_seconds-tp,'missed_seconds':actual_seconds-tp,'time_precision':tp/pred_seconds if pred_seconds else None,'time_recall':tp/actual_seconds if actual_seconds else None})
    tp=sum(x['true_positive_seconds'] for x in rows);fp=sum(x['false_positive_seconds'] for x in rows);fn=sum(x['missed_seconds'] for x in rows)
    result={'model':'media-sec-lab/BAM, official WavLM checkpoint','code_revision':'55f3fb9e3b4dd6281597b86d7712fb23454179f6','runtime':'CPU; native Microsoft WavLM via inference adapter; all checkpoint keys strict match; not yet equivalence-validated against original s3prl runtime','method':'0.16s frames, 10.24s independent chunks, fake label 0, fixed softmax threshold 0.5; no threshold tuning; chunk edges may affect scores','summary':{'replacement_cases':sum(bool(r['synthetic_intervals']) for r in rows),'overlap_hits':sum(r['overlap_hit'] is True for r in rows),'unchanged_controls':sum(not r['synthetic_intervals'] for r in rows),'controls_flagged':sum(not r['synthetic_intervals'] and bool(r['predicted_intervals']) for r in rows),'time_precision':tp/(tp+fp) if tp+fp else None,'time_recall':tp/(tp+fn) if tp+fn else None,'false_positive_seconds':fp,'runtime_seconds':time.monotonic()-t},'rows':rows}
    (out/'BAM检测结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
