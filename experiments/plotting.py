import io
import logging
import os
from typing import Optional, List

import PIL
import librosa
import numpy as np
import torch as tr
import torchaudio
from matplotlib import pyplot as plt
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from torch import Tensor as T
from torchaudio.transforms import Spectrogram, Fade
from torchvision.transforms import ToTensor

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def fig2img(fig: Figure, format: str = "png", dpi: int = 120) -> T:
    """Convert a matplotlib figure to tensor."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image


def plot_scalogram(ax: Subplot,
                   scalogram: T,
                   sr: float,
                   y_coords: List[float],
                   title: Optional[str] = None,
                   hop_len: int = 1,
                   cmap: str = "magma",
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   x_label: str = "time (s)",
                   y_label: str = "freq (Hz)",
                   fontsize: int = 12) -> None:
    """
    Plots a scalogram of the provided data.

    The scalogram is a visual representation of the wavelet transform of a signal over time.
    This function uses matplotlib and librosa to create the plot.

    Parameters:
        ax (Subplot): The axis on which to plot the scalogram.
        scalogram (T): The scalogram data to be plotted.
        sr (float): The sample rate of the audio signal.
        y_coords (List[float]): The y-coordinates for the scalogram plot.
        title (str, optional): The title of the plot. Defaults to "scalogram".
        hop_len (int, optional): The hop length for the time axis (or T). Defaults to 1.
        cmap (str, optional): The colormap to use for the plot. Defaults to "magma".
        vmax (Optional[float], optional): The maximum value for the colorbar. If None,
            the colorbar scales with the data. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to "Time (seconds)".
        y_label (str, optional): The label for the y-axis. Defaults to "Frequency (Hz)".
        fontsize (int, optional): The fontsize for the labels. Defaults to 16.
    """
    assert scalogram.ndim == 2
    assert scalogram.size(0) == len(y_coords)
    if vmin is not None and vmax is not None:
        # This is supposed to prevent an IndexError when vmin == vmax == 0.0
        if vmin == vmax:
            vmin = None
            vmax = None
        else:
            assert vmin < vmax
    x_coords = librosa.times_like(scalogram.size(1), sr=sr, hop_length=hop_len)

    try:
        librosa.display.specshow(ax=ax,
                                 data=scalogram.numpy(),
                                 sr=sr,
                                 x_axis="time",
                                 x_coords=x_coords,
                                 y_axis="cqt_hz",
                                 y_coords=np.array(y_coords),
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax)
    except IndexError as e:
        # TODO(cm): fix this
        log.warning(f"IndexError: {e}")

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    if len(y_coords) < 12:
        ax.set_yticks(y_coords)
    ax.minorticks_off()
    if title:
        ax.set_title(title, fontsize=fontsize)


def plot_spectrogram(audio: T,
                     ax: Optional[Subplot] = None,
                     title: Optional[str] = None,
                     save_name: Optional[str] = None,
                     save_dir: str = OUT_DIR,
                     sr: float = 44100,
                     fade_n_samples: int = 64) -> T:
    assert audio.ndim < 3
    audio = audio.detach()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    assert audio.size(0) == 1
    spectrogram = Spectrogram(n_fft=512, power=1, center=True, normalized=False)
    spec = tr.log(spectrogram(audio).squeeze(0))
    if ax is None:
        plt.imshow(spec, aspect="auto", interpolation="none")
        plt.title(title)
        plt.show()
    else:
        ax.imshow(spec, aspect="auto", interpolation="none")
        if title is not None:
            ax.set_title(title)

    if save_name is not None:
        sr = int(sr)
        if not save_name.endswith(".wav"):
            save_name = f"{save_name}.wav"
        if fade_n_samples:
            transform = Fade(fade_in_len=fade_n_samples, fade_out_len=fade_n_samples, fade_shape="linear")
            audio = transform(audio)
        save_path = os.path.join(save_dir, save_name)
        torchaudio.save(save_path, audio, sr)

    return spec


def plot_waveforms_stacked(waveforms: List[T],
                           sr: float,
                           title: Optional[str] = None,
                           waveform_labels: Optional[List[str]] = None,
                           show: bool = False) -> Figure:
    assert waveforms
    if waveform_labels is None:
        waveform_labels = [None] * len(waveforms)
    assert len(waveform_labels) == len(waveforms)

    fig, axs = plt.subplots(
        nrows=len(waveforms),
        sharex="all",
        sharey="all",
        figsize=(7, 2 * len(waveforms)),
        squeeze=False,
    )
    axs = axs.squeeze(1)

    for idx, (ax, w, label) in enumerate(zip(axs, waveforms, waveform_labels)):
        assert 0 < w.ndim <= 2
        if w.ndim == 2:
            assert w.size(0) == 1
            w = w.squeeze(0)
        w = w.detach().float().cpu().numpy()
        if idx == len(waveforms) - 1:
            axis = "time"
        else:
            axis = None
        librosa.display.waveshow(w, axis=axis, sr=sr, label=label, ax=ax)
        ax.set_title(label)
        ax.grid(color="lightgray", axis="x")
        # ax.set_xticks([])
        # ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    # fig.savefig(os.path.join(OUT_DIR, f"3.svg"))

    if show:
        fig.show()
    return fig


def plot_xy_points_and_grads(ax: Subplot,
                             x: T,
                             y: T,
                             x_hat: T,
                             y_hat: T,
                             x_grad: Optional[T] = None,
                             y_grad: Optional[T] = None,
                             title: str = "",
                             x_label: str = "θ slope",
                             y_label: str = "θ density",
                             x_min: float = -1.0,
                             x_max: float = 1.0,
                             y_min: float = 0.0,
                             y_max: float = 1.0,
                             fontsize: int = 12):
    for idx in range(len(x)):
        ax.plot([x_hat[idx], x[idx]],
                [y_hat[idx], y[idx]],
                color="lightgrey",
                linestyle="dashed",
                linewidth=1,
                zorder=0)
    if x_grad is not None and y_grad is not None:
        ax.quiver(x_hat.numpy(),
                  y_hat.numpy(),
                  -x_grad.numpy(),
                  -y_grad.numpy(),
                  color="red",
                  angles="xy",
                  scale=5.0,
                  scale_units="width",
                  zorder=1)
    ax.scatter(x_hat, y_hat, color="black", marker="o", facecolors='none', zorder=2)
    ax.scatter(x, y, color="black", marker="x", zorder=2)
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)


def plot_wavelet_2d(psi_2d: T, title: str = "wavelet 2D (time domain)", cmap: str = "inferno", plot_3d: bool = True) -> None:
    """
    :param psi_2d: complex tensor of wavelet coefficients in the time domain
    :param title: title of the plot
    :param cmap: colormap to use
    :param plot_3d: whether to plot the wavelet in 3D or not
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(psi_2d.real, cmap=cmap, interpolation='none', origin="lower", aspect=1)
    plt.xlabel("time", fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_xticks([])
    plt.ylabel("frequency", fontsize=18)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    plt.title(title, fontsize=18)
    plt.show()
    if plot_3d:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        t_y, t_x = tr.meshgrid(tr.arange(psi_2d.size(0)), tr.arange(psi_2d.size(1)))
        ax.plot_surface(t_x, t_y, psi_2d.real, cmap=cmap, antialiased=True, linewidth=0)
        ax.view_init(20, -80, 0)
        ax.set_xlabel("time", fontsize=18)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_xticks([])
        ax.set_ylabel("frequency", fontsize=18)
        ax.yaxis.set_tick_params(labelbottom=False)
        ax.set_yticks([])
        ax.set_zlabel("amplitude", fontsize=18)
        ax.zaxis.set_tick_params(labelbottom=False)
        ax.set_zticks([])
        plt.title(title, fontsize=18)
        plt.show()
