import hashlib,json
from huggingface_hub import snapshot_download
from core import ROOT,MODEL_ID,REVISION
MODELS=[('detector',MODEL_ID,REVISION,'905e330265c40a76955911ce86691fb4c8f2ed9c8085f4f046cac1383592a0ac'),('comparison','mo-thecreator/Deepfake-audio-detection','e4d9874b493362149cec96ced85f00b00b1a04c0','6eaf9d5638b6e32ffa93ba784523d664d37d4105021e83dedcdd5f99a2505f25')]
provenance=[]
for directory,repo,revision,expected in MODELS:
    p=ROOT/'models'/directory
    snapshot_download(repo,revision=revision,local_dir=p,allow_patterns=['config.json','preprocessor_config.json','model.safetensors','README.md'])
    files={}
    for f in p.iterdir():
        if f.is_file():
            h=hashlib.sha256()
            with f.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
            files[f.name]=h.hexdigest()
    if files['model.safetensors']!=expected:
        raise RuntimeError(f'{repo} SHA256 mismatch; remove incomplete file and retry.')
    provenance.append({'repo':repo,'revision':revision,'license':'Apache-2.0','sha256':files})
(ROOT/'models'/'provenance.json').write_text(json.dumps(provenance,indent=2))
print('Both models downloaded and hashes verified.')
