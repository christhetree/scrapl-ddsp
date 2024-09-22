import json
import logging
import os

import torch
import torchaudio
from matplotlib import pyplot as plt
from torch.distributions import Normal

from experiments.paths import CONFIGS_DIR, MODELS_DIR, DATA_DIR
from flowtron.data import Data
from flowtron.flowtron import Flowtron
# from flowtron.tacotron2.waveglow.denoiser import Denoiser
from flowtron.train import update_params

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    config_path = os.path.join(CONFIGS_DIR, "flowtron/config.json")
    model_path = os.path.join(MODELS_DIR, "flowtron_ljs.pt")
    dataset_path = os.path.join(DATA_DIR, "surprised_samples/surprised_audiofilelist_text.txt")

    params = ["model_config.dummy_speaker_embedding=0", "data_config.p_arpabet=1.0"]

    with open(config_path) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, params)

    data_config = config["data_config"]
    model_config = config["model_config"]

    state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    model = Flowtron(**model_config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # waveglow_path = "models/waveglow_256channels_universal_v5.pt"
    # waveglow = torch.load(waveglow_path, map_location="cpu")["model"]
    # waveglow.eval()
    # denoiser = Denoiser(waveglow)
    # denoiser.eval()

    dataset = Data(
        dataset_path,
        **dict(
            (k, v)
            for k, v in data_config.items()
            if k not in ["training_files", "validation_files"]
        ),
    )

    z_values = []
    force_speaker_id = 0
    for i in range(len(dataset)):
        mel, sid, text, attn_prior = dataset[i]
        mel, sid, text = mel[None], sid, text[None]
        if force_speaker_id > -1:
            sid = sid * 0 + force_speaker_id
        in_lens = torch.LongTensor([text.shape[1]])
        out_lens = torch.LongTensor([mel.shape[2]])
        with torch.no_grad():
            z = model(mel, sid, text, in_lens, out_lens)[0]
            log.info(f"mel.shape = {mel.shape}, z.shape: {z.shape}")
            z_values.append(z.permute(1, 2, 0))

    # Compute the posterior distribution
    lambd = 0.0001
    sigma = 0.01
    n_frames = 300
    # aggregation_type = "time_and_batch"
    aggregation_type = "batch"

    if aggregation_type == "time_and_batch":
        z_mean = torch.cat([z.mean(dim=2) for z in z_values])
        z_mean = torch.mean(z_mean, dim=0)[:, None]
        ratio = len(z_values) / lambd
        mu_posterior = ratio * z_mean / (ratio + 1)
    elif aggregation_type == "batch":
        for k in range(len(z_values)):
            expand = z_values[k]
            while expand.size(2) < n_frames:
                expand = torch.cat((expand, z_values[k]), 2)
            z_values[k] = expand[:, :, :n_frames]

        z_mean = torch.mean(torch.cat(z_values, dim=0), dim=0)[None]
        z_mean_size = z_mean.size()
        z_mean = z_mean.flatten()
        ratio = len(z_values) / float(lambd)
        mu_posterior = (ratio * z_mean / (ratio + 1)).flatten()
        mu_posterior = mu_posterior.view(80, -1)
    else:
        raise ValueError(f"Invalid aggregation type: {aggregation_type}")

    torch.save(mu_posterior, "../out/z_80_surprised_2.pt")
    exit()

    log.info(f"ratio = {ratio}, mu_posterior.shape = {mu_posterior.shape}")
    dist = Normal(mu_posterior.cpu(), sigma)

    z_baseline = torch.FloatTensor(1, 80, n_frames).normal_() * sigma
    if aggregation_type == "time_and_batch":
        z_posterior = dist.sample([n_frames]).permute(2, 1, 0)
    elif aggregation_type == "batch":
        z_posterior = dist.sample().view(1, 80, -1)[..., :n_frames]

    text = "Humans are walking on the streets?"
    text_encoded = dataset.get_text(text)[None]

    speaker = 0
    speaker_id = torch.LongTensor([speaker])
    with torch.no_grad():
        mel_posterior = model.infer(z_posterior, speaker_id, text_encoded)[0]
        mel_baseline = model.infer(z_baseline, speaker_id, text_encoded)[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 6))
    axes[0, 0].imshow(
        mel_posterior[0].cpu(), aspect="auto", origin="lower", interpolation="none"
    )
    im = axes[0, 1].imshow(
        z_posterior[0].cpu(), aspect="auto", origin="lower", interpolation="none"
    )
    plt.colorbar(im, ax=axes[0, 1])
    axes[1, 0].imshow(
        mel_baseline[0].cpu(), aspect="auto", origin="lower", interpolation="none"
    )
    im = axes[1, 1].imshow(
        z_baseline[0].cpu(), aspect="auto", origin="lower", interpolation="none"
    )
    plt.colorbar(im, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig("out/visualization.png")

    with torch.no_grad():
        audio = waveglow.infer(mel_posterior, sigma=0.75)

    sr = data_config["sampling_rate"]
    for idx, a in enumerate(audio):
        torchaudio.save(f"out/posterior_{idx}_sig_{sigma:.2f}.wav", a, sr)

    with torch.no_grad():
        audio = waveglow.infer(mel_baseline, sigma=0.75)
    for idx, a in enumerate(audio):
        torchaudio.save(f"out/baseline_{idx}_sig_{sigma:.2f}.wav", a, sr)
