# 语音取证工作台

自部署 Demo：分别检查 AI 合成／声音克隆迹象与剪辑／拼接的声学线索。支持 WAV、MP3、M4A、FLAC、OGG，双声道分别分析，提供波形、频谱、时间轴试听和 JSON 报告。

## 技术栈与目录

- 应用：Python 3.11、Streamlit 单进程本地网页；Plotly 展示波形与频谱，无独立前端或数据库服务。
- 音频：系统 FFmpeg/ffprobe 解码与元信息，NumPy、SciPy、SoundFile 做信号处理。
- 推理：PyTorch、Transformers、Hugging Face Hub；Wav2Vec2 双模型检测，faster-whisper / CTranslate2 本地中文转写。检测器支持 MPS/CUDA/CPU；转写当前固定 CPU int8。
- 可选 API：Requests 调 DeepSeek，只发送用户勾选同意的转写文本。
- 存储：本地文件、SHA-256 目录与 JSON 报告；无数据库。依赖见 `requirements.txt`，当时环境快照见 `requirements.lock.txt`，跨平台安装需重新核实兼容性。
- 入口：`app.py`；处理与模型逻辑：`core.py`；权重准备：`download_models.py`、`prepare_whisper.py`；局部合成实验：`experiments/bam/`。

## 启动

需要 Python 3.11、uv 和系统 ffmpeg（macOS 可用 `brew install ffmpeg`）。

```sh
cd ~/Git/audio-forensics-demo
./start.sh
```

打开 http://127.0.0.1:8502 。首次下载约 1.65 GB 双模型权重，之后可离线运行核心检测。关闭服务：在终端按 Ctrl-C。当前目录 `.venv` 已安装运行依赖。

## 实际运行的检测

- 合成检测：Wav2Vec2-XLSR 微调模型 `garystafford/wav2vec2-deepfake-voice-detector`，Apache-2.0，固定 revision `c66306024a7ede0be291e9c4558b37634782dc4e`。不是 AASIST；本版选用有 safetensors 权重与标准 Transformers 推理接口的模型。无第三方远程推理、无 trust_remote_code。
- 16 kHz，每声道独立 8 秒分段，短尾段合并到上一段；少于 2.5 秒或音量过低时不判定。softmax fake 类 >=0.8 标可疑，<=0.2 标未发现明显迹象；阈值未校准，分数不是真伪概率。支持 Apple MPS、CUDA、CPU。
- 剪辑线索：比较切点前后持续音量及频谱变化，排除显著静音边界，再做邻近点抑制。**这是工程规则，不是经过训练的剪辑检测模型**，可能把换人讲话、线路变化当作切点，也可能漏掉无缝拼接。
- Whisper small 在本地转写，权重必须预先准备，不再在点击转写后隐式下载。运行 `.venv/bin/python prepare_whisper.py` 会显示下载进度并校验 SHA256；模型就绪后完全离线加载，缓存模型并逐段显示识别进度。可选 DeepSeek 仅分析转写文本中的矛盾与缺失上下文，需提供 Key 并勾选允许外发文本。音频不发送到 DeepSeek。
- 不含特定人物声纹认证与专门重放检测。

模型作者微调数据主要为英文（1,866 个片段）。作者公布的验证集成绩不代表中文热线效果，尤其不能直接用于处罚或定责。下一阶段需要真实中文原件、克隆版本、拼接版本和电话编码／降噪对照，统计误报率、召回率、切点定位误差。

## 双模型对照与已发现的局限

主模型已将完整 LibriSpeech 真人朗读误报为合成（fake 分数 0.978），电话编码版本为 0.828。它也对系统 TTS 给出高分；这不能证明它具有可靠泛化能力。原始失败结果完整保留在 `validation-primary-only.json`。

增加独立权重 `mo-thecreator/Deepfake-audio-detection`（Apache-2.0，revision `e4d9874b493362149cec96ced85f00b00b1a04c0`）并列输出。该模型训练数据披露不足，也不能用作者准确率作为业务效果。两者均 >=0.8 才标可疑，均 <=0.2 才标未发现明显迹象，其余无法判断；这是未经独立评估的演示共识规则，不保证准确率改善。模型类别顺序不同，代码按各自 config 的 fake 标签取值，不能硬编码同一个类别索引。

最终双模型样例结果：中文系统 TTS 两段分别约为 (0.980, 1.000)、(0.982, 1.000)，共识为合成可疑；完整真人朗读约 (0.978, 0.000)，电话编码真人朗读约 (0.828, 0.0005)，均因模型分歧输出无法判断。人工拼接的真人语音也输出无法判断；合成检测不能代替剪辑鉴别。完整原始结果见 `validation-results.json`，它是功能对照，不是独立准确率评估。Whisper small 权重已完成 SHA256 校验，并在本地成功转写一段约 22 秒中文录音，实测约 3 秒；DeepSeek 未使用用户凭证发起实测请求。

首次对照中，规则在完整真人朗读上误报两个切点；人工拼接的真实切点 5.0 秒没有在 ±0.5 秒内定位到。因此剪辑模块仅用于复核导航，尚不具备可靠鉴别剪辑的能力。界面与导出报告均明确提示，不能以“能画切点”等同于“能识别造假”。

## 数据与报告

`data/<SHA256>/` 保存原件、16 kHz 解码副本、基础检测报告。下载报告额外包含当次转写与文本复核（如运行）。API Key 不写入报告。

转写结果另存为同目录 `transcript.json`，页面重启后可通过“恢复已保存的录音与分析结果”继续查看。中文转写默认使用中文，也可切换为自动识别或英语。

每个文件最多解码前 60 分钟，界面及报告明确该范围；原件完整保留。报告中的时间对应解码后的时间轴。录音文件及 models 均被 .gitignore 排除。

## 验证

```sh
.venv/bin/python -m pytest -q
.venv/bin/python validate.py
```

`test_core.py` 检查原件 SHA、双声道保留、已知声学突变定位和未运行模型时不输出真假结论。`validate.py` 在样例上运行真实权重并记录结果。

测试素材的来源和标签见 `samples/manifest.json`，它们只用于功能和局限验证，不能据此计算业务准确率。系统 TTS 不是某个真实人物的克隆。真实朗读样例与模型训练域可能存在重叠，不是独立盲测。

来源：
- https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector
- https://github.com/SYSTRAN/faster-whisper
- https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy

模型源文件、许可证声明与哈希保存在 `models/detector/README.md` 和 `models/provenance.json`。本项目界面及规则代码为演示实现。

## 局部合成替换实验（2026-10-16）

已接入可独立运行的 BAM 实验模块，尚未接入网页默认判定。用途是真人录音中局部合成语音定位，不代表能鉴别纯真人删除拼接。运行方法、权重位置、源码版本和授权限制见 [实验说明](experiments/bam/README.md) 与 `experiments/bam/provenance.json`。

两份中文热线各替换 1、3、6 秒系统普通话 TTS，另留两份未改动对照：旧双模型共识覆盖 0/6；BAM 覆盖 6/6，但两个对照都被标记，时间精确率约 10.6%、召回约 95.2%，区间外标记合计 160.8 秒。覆盖不等于精确定位，不能用于自动验收。本轮仅一个系统 TTS，引擎并非人物克隆；未改动对照也不保证历史真实性。没有依照本批答案调阈值。

## 版本管理状态

项目远端为私有仓库 `DanZai233/audio-forensics-demo`。`models/`、`data/`、环境和密钥不应提交；对外发布前还需检查样例、报告、截图中的业务信息及素材授权。

仓库仅包含代码、依赖和文档。真实案例、截图、生成报告、样例媒体与模型大文件保留在本机，不随 Git 分发；克隆后需另行准备权重和有权使用的测试素材。
