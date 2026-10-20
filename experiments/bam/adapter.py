"""Inference-only WavLM adapter for BAM, using Microsoft's upstream WavLM.

Avoids loading s3prl training dependencies. All checkpoint keys must match.
This adapter requires validation against the original runtime before production.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from wavlm.WavLM import WavLM, WavLMConfig


class WavLMUpstream(nn.Module):
    def __init__(self):
        super().__init__()
        # Microsoft wavlm-large config: cross-check microsoft/wavlm-large/config.json.
        self.cfg = WavLMConfig(dict(encoder_layers=24, encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096, encoder_attention_heads=16,
            extractor_mode='layer_norm', layer_norm_first=True,
            normalize=True, relative_position_embedding=True,
            gru_rel_pos=True, num_buckets=320, max_distance=800,
            feature_grad_mult=0., encoder_layerdrop=0.))
        self.model = WavLM(self.cfg)

    def forward(self, x):
        if self.cfg.normalize:
            x = F.layer_norm(x, (x.shape[-1],))
        features, _ = self.model.extract_features(x, mask=False)
        return {'hidden_states': [features]}


def wavlm_local(**kwargs):
    # The full fine-tuned SSL weights are expected in the BAM checkpoint.
    return WavLMUpstream()
