import logging
import os

import torchaudio
import torch as tr
from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2 = FastSpeech2.from_hparams(
    source="speechbrain/tts-fastspeech2-ljspeech",
    savedir="pretrained_models/tts-fastspeech2-ljspeech",
)
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
)

# Run TTS with text input
# input_text = "were the leaders in this luckless change; though our own Baskerville; who was at work some years before them; went much on the same lines;"
input_text = "Pigs are flying!"

pace = 1.0
pitch_rate = 1.0
energy_rate = 0.1

mel_output, durations, pitch, energy = fastspeech2.encode_text(
    [input_text],
    pace=pace,  # scale up/down the speed
    pitch_rate=pitch_rate,  # scale up/down the pitch
    energy_rate=energy_rate,  # scale up/down the energy
)
log.info(f"mel_output.shape: {mel_output.shape}")
# mel_output2, _, _, _ = fastspeech2.encode_text(
#     [input_text],
#     pace=pace,  # scale up/down the speed
#     pitch_rate=pitch_rate,  # scale up/down the pitch
#     energy_rate=energy_rate,  # scale up/down the energy
# )
# assert tr.allclose(mel_output, mel_output2)

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save(
    f"example_TTS_input_text_pace_{pace:.2f}_pitch_{pitch_rate:.2f}_energy_{energy_rate:.2f}.wav", waveforms.squeeze(1), 22050
)
