import hashlib, json, os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import signal
from core import ROOT, SR, ingest, analyze, load_detector, transcribe, explain, transcription_ready

st.set_page_config(page_title='语音取证工作台',layout='wide')
st.title('语音取证工作台')
st.caption('AI 合成／声音克隆检测 · 剪辑线索 · 分段试听 · 本地报告')
st.warning('研究演示：中文热线与电话压缩条件尚未校准。可疑分数和切点需要人工复核，不能据此认定造假。')
@st.cache_resource
def detector(): return {'primary':load_detector(), 'comparison':load_detector('comparison')}

with st.sidebar:
    st.header('分析设置')
    run_ai=st.checkbox('运行本地合成语音模型',value=True)
    st.caption('双模型权重约 1.65 GB；8 秒分段，双声道分别处理。每个文件最多分析前 60 分钟。')
    st.markdown('模型：[Wav2Vec2 检测器](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)')
    key=st.text_input('DeepSeek API Key（可选）',type='password',value=os.environ.get('DEEPSEEK_API_KEY',''))
    consent=st.checkbox('允许将转写文本发送到 DeepSeek',value=False)
    st.caption('原始音频不发送给 DeepSeek。转写使用本地 Whisper，不会在点击后隐式下载模型。')

upload=st.file_uploader('上传录音（WAV / MP3 / M4A / FLAC / OGG）',type=['wav','mp3','m4a','flac','ogg'])
samples=sorted((ROOT/'samples').glob('*.wav'))
selection=st.selectbox('或者选择测试素材', ['不选择']+[p.name for p in samples])
source=upload.getvalue() if upload else ((ROOT/'samples'/selection).read_bytes() if selection!='不选择' else None)
name=upload.name if upload else selection
saved_folder=None
if source is None:
    saved=sorted((ROOT/'data').glob('*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:20]
    if saved:
        labels=['不恢复']+[json.loads(p.read_text())['input']['filename']+' · '+p.parent.name[:8] for p in saved]
        chosen=st.selectbox('恢复已保存的录音与分析结果',labels,index=1)
        if chosen!='不恢复':
            saved_folder=saved[labels.index(chosen)-1].parent
            report=json.loads((saved_folder/'report.json').read_text())
            original=next(saved_folder.glob('original.*'))
            source=original.read_bytes()
            name=report['input']['filename']
if not source:
    st.info('上传真实录音即可检查。测试素材包含公开真人朗读、系统 TTS 和已知拼接对照，不代表中文热线的实际准确率。')
    st.stop()
if not upload and saved_folder is None and (ROOT/'samples'/'manifest.json').exists():
    manifest=json.loads((ROOT/'samples'/'manifest.json').read_text())
    st.caption('测试素材标签：'+json.dumps(manifest.get(selection,{}),ensure_ascii=False))
identity=hashlib.sha256(source).hexdigest()
if st.session_state.get('identity')!=identity:
    for k in ['audio','meta','folder','report','transcript','explanation']:
        st.session_state.pop(k,None)
    st.session_state.identity=identity
if saved_folder is not None and 'report' not in st.session_state:
    import soundfile as sf
    audio,_=sf.read(saved_folder/'decoded.wav',dtype='float32',always_2d=True)
    st.session_state.update(audio=audio,meta=report['input'],folder=saved_folder,report=report)
    if (saved_folder/'transcript.json').exists():
        st.session_state.transcript=json.loads((saved_folder/'transcript.json').read_text())
weights=ROOT/'models'/'detector'/'model.safetensors'
other_weights=ROOT/'models'/'comparison'/'model.safetensors'
ready=weights.exists() and weights.stat().st_size==1262859296 and other_weights.exists() and other_weights.stat().st_size==378302360
if run_ai and not ready: st.info('合成检测权重尚未准备完成，请等待下载或运行 python download_models.py。')
if st.button('开始检测',type='primary',disabled=run_ai and not ready):
    try:
        with st.spinner('解码并加载本地模型…'):
            audio,meta,folder=ingest(source,name)
            st.session_state.update(audio=audio,meta=meta,folder=folder)
            model=detector() if run_ai else None
        bar=st.progress(0.,text='按声道分析录音…')
        report=analyze(audio,meta,model,lambda v:bar.progress(v))
        st.session_state.report=report
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        bar.empty()
    except Exception as e:
        st.error(f'分析失败：{e}')
if 'report' not in st.session_state: st.stop()
r=st.session_state.report
x=st.session_state.audio
st.subheader('合成检测：'+r['synthetic_verdict'])
st.warning('主模型已在完整真人朗读上出现高分误报。两模型均有强响应才标可疑，分歧时无法判断；此共识规则尚未独立评估，同意也不代表正确。')
st.info('剪辑线索模块处于实验阶段：已在未剪辑真人朗读上出现误报，并漏掉已知拼接点。切点仅作试听导航，不代表已检测到剪辑。')
if not r['model']['ran']: st.info('本次未运行合成模型，仅分析剪辑线索。')
a,b,c=st.columns(3)
a.metric('分析时长',f"{r['input']['duration']:.1f} 秒")
b.metric('声道数',r['input']['channels'])
c.metric('可疑声学切点',sum(len(ch['cuts']) for ch in r['channels']))
st.caption('SHA256：'+r['input']['sha256'])
streams=r['input'].get('source_metadata',{}).get('streams',[])
if any(int(s.get('sample_rate',16000))<=8000 for s in streams):
    st.warning('源录音为 8 kHz 或更低采样率：电话编码可能影响合成检测分数，升采样不会恢复丢失信息。')
st.audio(source)
for ch in r['channels']:
    i=ch['channel']-1
    with st.expander(f'声道 {i+1}：时间轴与复核',expanded=True):
        stride=max(1,len(x)//12000)
        fig=go.Figure(go.Scatter(x=np.arange(0,len(x),stride)/SR,y=x[::stride,i],name='波形'))
        for cut in ch['cuts']: fig.add_vline(x=cut['time'],line_color='orange',line_dash='dash')
        fig.update_layout(height=220,xaxis_title='秒',yaxis_title='幅度',margin=dict(t=15,b=20))
        st.plotly_chart(fig,width='stretch')
        if ch['synthetic']:
            st.dataframe(ch['synthetic'],hide_index=True,width='stretch')
            st.caption('score 为主模型 fake 类输出，score_comparison 为对照模型；均 ≥0.8 才标可疑，均 ≤0.2 才标未发现明显迹象，其余无法判断。分数不等于造假概率。')
        if ch['cuts']:
            st.dataframe(ch['cuts'],hide_index=True,width='stretch')
            t=st.selectbox('选择切点并试听前后各 2 秒',[p['time'] for p in ch['cuts']],key=f'cut{i}')
            st.audio(x[max(0,int((t-2)*SR)):int((t+2)*SR),i],sample_rate=SR)
        else: st.caption('未发现满足当前规则的声学突变；无法排除无缝剪辑。')
        if st.checkbox('查看频谱图',key=f'spec{i}'):
            # Display at most the first 120 seconds to bound memory.
            f,t,z=signal.stft(x[:120*SR,i],fs=SR,nperseg=512,noverlap=256)
            fig=go.Figure(go.Heatmap(x=t[::4],y=f,z=20*np.log10(np.abs(z[:,::4])+1e-7),colorscale='Viridis',zmin=-90,zmax=-20))
            fig.update_layout(height=260,xaxis_title='秒（前120秒）',yaxis_title='Hz')
            st.plotly_chart(fig,width='stretch')
        start=st.number_input('试听起点（秒）',0.,max(0.,len(x)/SR-.1),0.,key=f'listen{i}')
        st.audio(x[int(start*SR):int((start+10)*SR),i],sample_rate=SR)

st.subheader('转写与内容复核（可选）')
language_label=st.selectbox('转写语言',['中文','自动识别','英语'])
language={'中文':'zh','自动识别':None,'英语':'en'}[language_label]
if not transcription_ready():
    st.warning('Whisper small 权重未准备好，请先在项目目录运行 .venv/bin/python prepare_whisper.py。')
else:
    st.caption('Whisper small 本地权重已就绪；转写不需要联网，长录音将逐段显示进度。')
if st.button('本地 Whisper 转写',disabled=not transcription_ready()):
    progress=st.progress(0.,text='准备本地转写…')
    try:
        st.session_state.transcript=transcribe(st.session_state.folder/'decoded.wav',lambda value,text:progress.progress(value,text=text),language=language)
    except Exception as e:
        progress.empty()
        st.error(f'转写失败：{e}')
if 'transcript' in st.session_state:
    st.dataframe(st.session_state.transcript['segments'],hide_index=True)
    if st.button('用 DeepSeek 核对文本内容',disabled=not (key and consent)):
        try:
            with st.spinner('核对文本中…'): st.session_state.explanation=explain(st.session_state.transcript,key)
        except Exception as e: st.error(f'DeepSeek 请求失败：{type(e).__name__}')
if 'explanation' in st.session_state: st.write(st.session_state.explanation)
export=dict(r)
if 'transcript' in st.session_state: export['transcript']=st.session_state.transcript
if 'explanation' in st.session_state: export['text_review']=st.session_state.explanation
st.download_button('下载 JSON 报告',json.dumps(export,ensure_ascii=False,indent=2),file_name='audio-report.json',mime='application/json')
with st.expander('结果局限与模型信息'):
    for item in r['limitations']: st.write('• '+item)
    st.json(r['model'])
