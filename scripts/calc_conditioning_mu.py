import json
import logging
import os

import pyloudnorm as pyln
import torch as tr
import torchaudio
from torch import Tensor as T
from tqdm import tqdm

from experiments.paths import OUT_DIR, CONFIGS_DIR, MODELS_DIR, DATA_DIR
from flowtron.audio_processing import TacotronSTFT
from flowtron.data import get_text_embedding
from flowtron.flowtron import Flowtron
from flowtron.text.cmudict import CMUDict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


idx_to_sentence = {
    1: "Kids are talking by the door.",
    2: "Dogs are sitting by the door.",
}
idx_to_emotion = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def peak_normalize(audio: T, peak_norm_db: float = -1.0) -> T:
    assert audio.ndim == 2
    audio_np = audio.T.numpy()
    audio_norm_np = pyln.normalize.peak(audio_np, peak_norm_db)
    audio_norm = tr.from_numpy(audio_norm_np.T)
    return audio_norm


def prepare_ravdess_data(
    root_dir: str, out_dir: str, gender: str, emotion: str
) -> None:
    save_dir = os.path.join(out_dir, f"{gender}_{emotion}")
    os.makedirs(save_dir, exist_ok=True)

    for actor_idx in tqdm(range(1, 25, 2)):
        if gender == "female":
            actor_idx += 1
        curr_dir_path = os.path.join(root_dir, f"Actor_{actor_idx:02d}")
        for file_name in os.listdir(curr_dir_path):
            if not file_name.endswith(".wav"):
                continue
            tokens = file_name.split("-")
            curr_emotion_idx = int(tokens[2])
            if idx_to_emotion[curr_emotion_idx] != emotion:
                continue
            audio, sr = torchaudio.load(os.path.join(curr_dir_path, file_name))

            # Trim silence at front
            audio = torchaudio.transforms.Vad(sample_rate=sr)(audio)
            # Trim silence at back
            audio = tr.flip(audio, dims=(-1,))
            audio = torchaudio.transforms.Vad(sample_rate=sr)(audio)
            audio = tr.flip(audio, dims=(-1,))

            # Resample audio to 22050 Hz
            out_sr = 22050
            audio = torchaudio.transforms.Resample(sr, out_sr)(audio)

            # Normalize loudness
            audio = peak_normalize(audio)

            save_path = os.path.join(save_dir, file_name)
            torchaudio.save(save_path, audio, out_sr)


if __name__ == "__main__":
    # ravdess_dir = os.path.join(DATA_DIR, "ravdess")
    # gender = "male"
    # # emotion = "surprised"
    # emotion = "neutral"
    # # emotion = "calm"
    # # emotion = "happy"
    # # emotion = "disgust"
    # # emotion = "fearful"
    # # emotion = "sad"
    # # emotion = "angry"
    # prepare_ravdess_data(ravdess_dir, OUT_DIR, gender, emotion)
    # exit()

    config_path = os.path.join(CONFIGS_DIR, "flowtron/config.json")
    model_path = os.path.join(MODELS_DIR, "flowtron_ljs.pt")
    # dataset_name = "female_angry_14"
    dataset_name = "female_angry_24"
    # dataset_name = "female_sad_22"
    # dataset_name = "female_fearful_24"
    # dataset_name = "female_surprised"
    # dataset_name = "female_surprised_14"
    # dataset_name = "female_surprised_24"
    # dataset_name = "female_disgust_24"
    # dataset_name = "female_happy_24"
    # dataset_name = "male_neutral_21"
    # dataset_name = "male_neutral_23"
    dataset_path = os.path.join(OUT_DIR, dataset_name)
    n_frames = 128

    with open(config_path, "r") as f:
        config = json.load(f)
    data_config = config["data_config"]
    model_config = config["model_config"]

    state_dict = tr.load(model_path, map_location="cpu")["state_dict"]
    model = Flowtron(**model_config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    text_cleaners = data_config["text_cleaners"]
    cmudict_path = data_config["cmudict_path"]
    assert os.path.isfile(cmudict_path)
    keep_ambiguous = data_config["keep_ambiguous"]
    cmudict = CMUDict(cmudict_path, keep_ambiguous=keep_ambiguous)

    stft = TacotronSTFT(
        filter_length=data_config["filter_length"],
        hop_length=data_config["hop_length"],
        win_length=data_config["win_length"],
        sampling_rate=data_config["sampling_rate"],
        mel_fmin=data_config["mel_fmin"],
        mel_fmax=data_config["mel_fmax"],
    )

    idx_to_text_emb_p0 = {
        1: get_text_embedding(
            idx_to_sentence[1], text_cleaners, cmudict, p_arpabet=0.0
        ),
        2: get_text_embedding(
            idx_to_sentence[2], text_cleaners, cmudict, p_arpabet=0.0
        ),
    }
    idx_to_text_emb_p1 = {
        1: get_text_embedding(
            idx_to_sentence[1], text_cleaners, cmudict, p_arpabet=1.0
        ),
        2: get_text_embedding(
            idx_to_sentence[2], text_cleaners, cmudict, p_arpabet=1.0
        ),
    }

    z_s = []
    for file_name in tqdm(os.listdir(dataset_path)):
        if not file_name.endswith(".wav"):
            continue
        audio, sr = torchaudio.load(os.path.join(dataset_path, file_name))
        assert sr == data_config["sampling_rate"]
        tokens = file_name.split("-")
        intensity_idx = int(tokens[3])
        sentence_idx = int(tokens[4])
        z_multiplier = intensity_idx / 2.0

        speaker_id = tr.tensor([0]).long()
        text_emb_p0 = idx_to_text_emb_p0[sentence_idx]
        # text_emb_p1 = idx_to_text_emb_p1[sentence_idx]
        mel = stft.mel_spectrogram(audio)
        log.info(f"mel.shape = {mel.shape}")
        out_lens = tr.tensor([mel.size(-1)]).long()

        # for text_emb in [text_emb_p0, text_emb_p1]:
        for text_emb in [text_emb_p0]:
        # for text_emb in [text_emb_p1]:
            text_emb = text_emb.unsqueeze(0)
            in_lens = tr.tensor([text_emb.size(-1)]).long()
            with tr.no_grad():
                z = model(mel, speaker_id, text_emb, in_lens, out_lens)[0]
                z = tr.swapaxes(z, 0, 2)
                z = tr.swapaxes(z, 0, 1)
                n_repeats = (n_frames // z.size(-1)) + 1
                z = z.repeat(1, 1, n_repeats)
                z = z[..., :n_frames]
                # z = z.mean(dim=0)
                # z *= z_multiplier
            z_s.append(z)

    log.info(f"len(z_s) = {len(z_s)}")
    z = tr.cat(z_s, dim=0)
    log.info(f"z.shape before mean = {z.shape}")
    z = z.mean(dim=0)
    log.info(f"z.shape after mean = {z.shape}")
    tr.save(z, os.path.join(OUT_DIR, f"z_{dataset_name}_tv_a0.pt"))
    # z = z.view(-1, 1).repeat(1, n_frames)
    # tr.save(z, os.path.join(OUT_DIR, f"z_{dataset_name}_a0.pt"))
