import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import soundfile
import torch
from hypy_utils.downloader import download_file
from torch import Tensor as T
from torch import nn

from experiments import panns

log = logging.getLogger(__name__)


class ModelLoader(ABC, nn.Module):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """

    def __init__(self, name: str, num_features: int, sr: int, audio_len: int):
        super().__init__()
        self.audio_len = audio_len
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        # self.device = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )

    def get_embedding(self, audio: T) -> T:
        assert audio.ndim == 2
        embd = self._get_embedding(audio)
        # if self.device == torch.device("cuda"):
        #     embd = embd.cpu()
        # embd = embd.detach().numpy()

        # # If embedding is float32, convert to float16 to be space-efficient
        # if embd.dtype == np.float32:
        #     embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: T) -> T:
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype="int16")
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        # Ensure the audio length is correct
        if self.audio_len is not None and wav_data.shape[0] != self.audio_len * self.sr:
            raise RuntimeError(
                f"Audio is too long ({wav_data.shape[0] / self.sr:.2f} seconds > {self.audio_len} seconds)."
                + f"\n\t- {wav_file}"
            )
        return wav_data


class PANNsModel(ModelLoader):
    """
    Kong, Qiuqiang, et al., "Panns: Large-scale pretrained audio neural networks for audio pattern recognition.",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

    Specify the model to use (cnn14-32k, cnn14-16k, wavegram-logmel).
    """

    def __init__(
        self,
        variant: Literal["cnn14-32k", "cnn14-16k", "wavegram-logmel"],
        audio_len=None,
    ):
        super().__init__(
            name=f"panns-{variant}",
            num_features=2048,
            sr=16000 if variant == "cnn14-16k" else 32000,
            audio_len=audio_len,
        )
        self.variant = variant

    def load_model(self):
        current_file_dir = os.path.dirname(os.path.realpath(__file__))
        ckpt_dir = os.path.join(current_file_dir, "panns/ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        features_list = ["2048", "logits"]
        current_file_dir = os.path.dirname(os.path.realpath(__file__))

        # map_location = "cuda" if torch.cuda.is_available() else "cpu"
        if self.variant == "cnn14-16k":
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
            if not os.path.isfile(os.path.join(ckpt_dir, "Cnn14_16k_mAP=0.438.pth")):
                download_file(
                    "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
                    os.path.join(ckpt_dir, "Cnn14_16k_mAP=0.438.pth"),
                )
            state_dict = torch.load(
                f"{current_file_dir}/panns/ckpt/Cnn14_16k_mAP=0.438.pth",
                map_location="cpu",
            )
            self.model.load_state_dict(state_dict["model"])

        elif self.variant == "cnn14-32k":
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            if not os.path.isfile(os.path.join(ckpt_dir, "Cnn14_mAP=0.431.pth")):
                download_file(
                    "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth",
                    os.path.join(ckpt_dir, "Cnn14_mAP=0.431.pth"),
                )
            state_dict = torch.load(
                f"{current_file_dir}/panns/ckpt/Cnn14_mAP=0.431.pth", map_location="cpu"
            )
            self.model.load_state_dict(state_dict["model"])

        elif self.variant == "wavegram-logmel":
            self.model = panns.Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            if not os.path.isfile(
                os.path.join(ckpt_dir, "Wavegram_Logmel_Cnn14_mAP=0.439.pth")
            ):
                download_file(
                    "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth",
                    os.path.join(ckpt_dir, "Wavegram_Logmel_Cnn14_mAP=0.439.pth"),
                )
            state_dict = torch.load(
                f"{current_file_dir}/panns/ckpt/Wavegram_Logmel_Cnn14_mAP=0.439.pth",
                map_location="cpu",
            )
            self.model.load_state_dict(state_dict["model"])

        else:
            raise ValueError(f"Unexpected variant of PANNs model: {self.variant}.")

        self.model.eval()
        # self.model.to(self.device)

    def _get_embedding(self, audio: T) -> T:
        # if len(audio.shape) == 1:
        #     audio = audio.unsqueeze(0)
        if "cnn14" in self.variant:
            emb = self.model.forward(audio)["2048"]
        else:
            emb = self.model.forward(audio)["embedding"]
        return emb
