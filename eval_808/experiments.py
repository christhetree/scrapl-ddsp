import logging
import os

import torch as tr
import torchaudio

from eval_808.features import (
    FeatureCollection,
    Loudness,
    SpectralCentroid,
    SpectralFlatness,
    CascadingFrameExtactor,
    TemporalCentroid,
)
from eval_808.synths import Snare808

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    sr = 44100
    # sr = 32768
    synth = Snare808(
        sample_rate=sr,
        num_samples=sr,
        buffer_noise=True,
        buffer_size=sr,
    )
    params = tr.full((3, 14), fill_value=0.5)
    x = synth(params)
    log.info(f"x.shape = {x.shape}")
    # torchaudio.save("snare808.wav", x[0:1, :].cpu(), sr)

    f1 = CascadingFrameExtactor(
        extractors=[
            Loudness(sr),
            SpectralCentroid(
                sample_rate=sr,
                window="flat_top",
                compress=True,
                floor=1e-4,
                scaling="kazazis",
            ),
            SpectralFlatness(),
        ],
        num_frames=[2, 64],
        frame_size=2048,
        hop_size=512,
    )
    f2 = CascadingFrameExtactor(
        extractors=[
            TemporalCentroid(sample_rate=sr, scaling="schlauch"),
        ],
        num_frames=[1],
        frame_size=sr // 8,
        hop_size=sr // 8,
    )
    features = FeatureCollection(features=[f1, f2])

    feats = features(x)

    log.info(f"feats.shape: {feats.shape}")
