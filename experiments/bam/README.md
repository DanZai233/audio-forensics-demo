# BAM 局部合成定位实验

独立于 Streamlit 默认管线，暂不用于自动验收。2026-10-16 实测：中文局部系统 TTS 替换区间覆盖 6/6，未改动对照被标记 2/2；时间精确率约 10.6%，召回约 95.2%，误报严重。

在项目根目录运行：

```sh
.venv/bin/python experiments/bam/evaluate.py --suite /绝对路径/audio-partial-synthetic-test
```

套件目录需包含 `测试音频/*.wav`（16kHz 单声道）和 `核对答案.json`，格式参照此次实验输出。结果保存到同一目录；固定阈值 0.5，10.24 秒分块、0.16 秒输出。分块边缘可能影响结果。先做预测，再读取标注计算指标。不将普通删除拼接视为已验证能力。

模型位于 `models/bam/model.ckpt`，来源、版本和 SHA256 见 provenance.json。使用 weights_only 安全加载与严格键匹配；推理用原版 WavLM 源码做轻量适配，未复现完整旧版训练环境。按参考 expert.py 的归一化、padding mask 和最终 encoder hook 复核前向，最大差为 0；这不等于业务效果验证。

BAM 整体授权未明确，商业接入前需联系作者。WavLM 源码保留 MIT 版权。
