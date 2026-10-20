import json,time
from core import ROOT,ingest,analyze,load_detector
model={'primary':load_detector(),'comparison':load_detector('comparison')}
results=[]
for p in sorted((ROOT/'samples').glob('*.wav')):
    start=time.monotonic()
    audio,meta,folder=ingest(p.read_bytes(),p.name)
    report=analyze(audio,meta,model)
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    results.append({'file':p.name,'elapsed_seconds':round(time.monotonic()-start,2),'report':report})
    print(p.name,report['synthetic_verdict'],flush=True)
(ROOT/'validation-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
