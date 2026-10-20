"""Prepare pinned Whisper weights explicitly; no downloads during transcription."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib, json, os, time, urllib.request

ROOT=Path(__file__).resolve().parent
REVISION='536b0662742c02347bc0e980a01041f333bce120'
BASE=f'https://huggingface.co/Systran/faster-whisper-small/resolve/{REVISION}/'
SIZE=483546902
EXPECTED='3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def prepare():
    folder=ROOT/'models'/'whisper-small'
    folder.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','tokenizer.json','vocabulary.txt','README.md']:
        if not (folder/name).exists():
            with urllib.request.urlopen(BASE+name,timeout=30) as r:
                (folder/name).write_bytes(r.read())
    target=folder/'model.bin'
    if target.exists() and target.stat().st_size==SIZE and sha(target)==EXPECTED:
        print('Whisper small already verified.')
        return
    partial=target.with_suffix('.partial')
    fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o644)
    os.ftruncate(fd,SIZE)
    chunk=8*1024*1024
    def get(i):
        start=i*chunk; end=min(start+chunk,SIZE)-1
        for attempt in range(4):
            try:
                req=urllib.request.Request(BASE+f'model.bin?part={i}&try={attempt}',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                        raise RuntimeError('Server did not honor range request')
                    data=r.read()
                if len(data)!=end-start+1: raise RuntimeError('Length mismatch')
                os.pwrite(fd,data,start)
                return
            except Exception:
                if attempt==3: raise
                time.sleep(1)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            jobs=[pool.submit(get,i) for i in range((SIZE+chunk-1)//chunk)]
            for n,f in enumerate(as_completed(jobs),1):
                f.result()
                print(f'Download {n}/{len(jobs)} chunks ({100*n/len(jobs):.1f}%)',flush=True)
    finally:
        os.close(fd)
    if sha(partial)!=EXPECTED: raise RuntimeError('SHA256 mismatch; incomplete model not activated.')
    partial.replace(target)
    (folder/'provenance.json').write_text(json.dumps({'repo':'Systran/faster-whisper-small','revision':REVISION,'license':'MIT','sha256':EXPECTED},indent=2))
    print('Whisper small ready. Transcription now works offline.')

if __name__=='__main__': prepare()
