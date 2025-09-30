import logging
import os

import pyloudnorm as pyln
import torch as tr
import torchaudio

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


if __name__ == "__main__":
    root_dir = os.path.join(OUT_DIR, "meso")
    # root_dir = os.path.join(OUT_DIR, "micro")
    peak_norm_db = -1.0

    # Find all .wav files
    for p in os.listdir(root_dir):
        if not p.endswith(".wav"):
            continue
        p = os.path.join(root_dir, p)
        log.info(f"Processing {p}")
        # Load audio
        audio, sr = torchaudio.load(p)
        n_ch = audio.size(0)
        assert sr == 44100
        assert n_ch == 1

        audio_np = audio.T.numpy()
        audio_norm_np = pyln.normalize.peak(audio_np, peak_norm_db)
        audio_norm = tr.from_numpy(audio_norm_np.T)
        torchaudio.save(p, audio_norm_np, sr, bits_per_sample=16)
