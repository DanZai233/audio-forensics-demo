import numpy as np
from core import SR, cut_clues, ingest, analyze
import io,soundfile as sf,hashlib

def test_constant_and_silence_are_not_edits():
    t=np.arange(SR*4)/SR
    assert cut_clues(np.zeros(SR*4))==[]
    assert cut_clues(.1*np.sin(2*np.pi*200*t))==[]

def test_known_acoustic_change_localized():
    t=np.arange(SR*3)/SR
    x=np.r_[.03*np.sin(2*np.pi*200*t),.3*np.sin(2*np.pi*2500*t)]
    cuts=cut_clues(x)
    assert cuts and abs(cuts[0]['time']-3)<.5

def test_preserves_stereo_and_original(tmp_path,monkeypatch):
    import core
    monkeypatch.setattr(core,'ROOT',tmp_path)
    t=np.arange(SR*3)/SR
    audio=np.stack([.1*np.sin(2*np.pi*200*t),.2*np.sin(2*np.pi*400*t)],axis=1)
    b=io.BytesIO();sf.write(b,audio,SR,format='WAV');raw=b.getvalue()
    decoded,meta,p=ingest(raw,'test.wav')
    assert meta['channels']==2 and decoded.shape==audio.shape
    assert (p/'original.wav').read_bytes()==raw
    assert meta['sha256']==hashlib.sha256(raw).hexdigest()
    report=analyze(decoded,meta)
    assert report['synthetic_verdict']=='无法判断'
    assert not report['model']['ran']


def test_disagreement_is_inconclusive():
    from core import consensus
    assert consensus(.99,.01)==consensus(.01,.99)
    assert consensus(.99,.99)!=consensus(.99,.01)
    assert consensus(.01,.01)!=consensus(.99,.01)
    assert consensus(None,.99)==consensus(.99,None)


def test_transcription_missing_weights_fails_without_download(tmp_path,monkeypatch):
    import core,pytest
    monkeypatch.setattr(core,'ROOT',tmp_path)
    core.load_transcriber.cache_clear()
    assert not core.transcription_ready()
    with pytest.raises(RuntimeError,match='prepare_whisper.py'):
        core.load_transcriber()
