import logging
import os

import numpy as np
import torch as tr
from matplotlib import pyplot as plt
from nnAudio.features import CQT

from experiments.plotting import plot_scalogram
from experiments.synths import ChirpletSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def gauss_window(M: float, std: tr.FloatTensor, sym: bool = True):
    """Gaussian window converted from scipy.signal.gaussian"""
    if M < 1:
        return tr.array([])
    if M == 1:
        return tr.ones(1, "d").type_as(std)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = tr.arange(0, M) - (M - 1.0) / 2.0
    n = n.type_as(std)

    sig2 = 2 * std * std
    w = tr.exp(-(n**2) / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def generate_am_chirp(
    theta: tr.FloatTensor,
    bw: float = 2,
    duration: float = 4,
    sr: float = 2**13,
    delta: int = 0,
):
    """
    Generate an amplitude-modulated chirp signal.
    The signal is windowed in the time domain by a Gaussian function, whose width
    is controlled by `bw` measured in octaves. All chirps cover a bandwidth `bw`.

    Args:
        theta (Union[tr.FloatTensor]): The carrier frequency (f_c) in Hz, modulator frequency (f_m) in Hz,
            and chirp rate (gamma) in octaves / second of the chirp signal.
        bw (float, optional): The bandwidth of the chirp signal in octaves. Defaults to 2.
        duration (float, optional): The duration of the chirp signal in seconds. Defaults to 4.
        sr (float, optional): The sample rate of the chirp signal in samples per second. Defaults to 2**13.
        delta (float, optional): applies a time shift in samples
    Returns:
        tr.FloatTensor: The generated amplitude-modulated chirp signal.

    Example:
        >>> theta = [tr.tensor(512.0), tr.tensor(8.0), tr.tensor(1.8)]
        >>> signal = generate_am_chirp(theta, bw=5, duration=2, sr=44100, delta=100)
        >>> signal.shape
    """
    f_c, f_m, gamma = theta[0], theta[1], theta[2]
    t = tr.arange(-duration / 2, duration / 2, 1 / sr).type_as(f_m)
    carrier = sine(f_c / (gamma * np.log(2)) * (2 ** (gamma * t) - 1))
    modulator = sine(t * f_m)
    sigma0 = 0.1
    window_std = (tr.tensor(sigma0 * bw).type_as(gamma)) / gamma
    window = gauss_window(duration * sr, std=window_std * sr)
    x = carrier * (modulator if f_m != 0 else 1.0) * window * float(gamma)
    if delta:
        x = time_shift(x, delta)
    return x


def sine(phi):
    return tr.sin(2 * tr.pi * phi)


def time_shift(x, delta):
    y = tr.zeros_like(x)
    y[delta:] = x[:-delta]
    return y


def chirp(t, gamma=0.5, f_c=512):
    chirp_phase = 2 * np.pi * f_c / (gamma * np.log(2)) * (2 ** (gamma * t) - 1)
    return np.sin(chirp_phase)


def am_sine(f_c, f_m, duration=2, sr=2**14):
    t = np.arange(-duration / 2, duration / 2, 1 / sr)
    carrier = np.sin(2 * np.pi * f_c * t)
    modulator = np.sin(2 * np.pi * f_m * t)
    x = carrier * modulator
    return x


def grid2d(x1: float, x2: float, y1: float, y2: float, n: float):
    a = tr.logspace(np.log10(x1), np.log10(x2), n)
    b = tr.logspace(np.log10(y1), np.log10(y2), n)
    X = a.repeat(n)
    Y = b.repeat(n, 1).t().contiguous().view(-1)
    return X, Y


def ripple(theta, duration, n_partials, sr, window=False):
    """Synthesizes a ripple sound.
    Args:
        theta: [v, w, f0, fm1]
            v (float): octaves per second, w / omega
            omega (float): amount of phase shift at each partial. (Ripple density)
            w (float): Amplitude modulation frequency in Hz. (Ripple drift)
            delta (float): Normalized ripple depth. Value must be in
                the range [0, 1].
            f0 (float): Frequency of the lowest sinusoid in Hz.
            fm1 (float): Frequency of the highest sinusoid in Hz.
        duration (float): Duration of sound in seconds.
        n_partials (int): Number of sinusoids.
        sr (int): Sampling rate in Hz.

    Returns:
        y (tr.tensor): The waveform.
    """
    omega, w, f0, fm1 = theta
    device = omega.device
    phi = 0.0
    # create sinusoids
    m = int(duration * sr)  # total number of samples
    t = tr.linspace(0, duration, int(m)).to(device).reshape((1, -1))
    i = tr.arange(n_partials).to(device).reshape((n_partials, 1))
    # space f0 and highest partial evenly in log domain (divided by # partials)
    f = f0 * (fm1 / f0) ** (i / (n_partials - 1))
    sphi = 0  # 2 * tr.pi * tr.rand((1, n_partials, 1)).to(device)
    s = tr.sin(2 * tr.pi * f * t + sphi)

    # create envelope
    x = tr.log2(f / f0)
    delta = 1.0
    # a = 1.0 + delta * tr.sin(
    #     2 * tr.pi * w[:, :, None] * (t + x / (v[:, :, None])) + phi
    # )
    a = 1.0 + delta * tr.sin(2 * tr.pi * (w * t + x * omega) + phi)
    # win = tr.hann_window(duration * sr) if window else 1.0
    # create the waveform, summing partials
    y = tr.sum(a * s, dim=0)  # / tr.sqrt(f)
    y = y / tr.sum(tr.abs(y))

    return y


if __name__ == "__main__":
    sr = 8192
    dur_sec = 4
    n_samples = int(dur_sec * sr)
    bw = 2.0
    bw_n_samples = int(bw * sr)
    f0_hz = 512
    am_hz = 1.0  # 4 to 16 Hz
    # fm_oct_hz = 1.414 # 0.5 to 4 octaves per second
    fm_oct_hz = 1.0 # 0.5 to 4 octaves per second
    delta = 0
    # delta = 16000
    # delta = -delta

    theta = tr.tensor([f0_hz, am_hz, fm_oct_hz])
    x = generate_am_chirp(theta, bw=bw, duration=dur_sec, sr=sr, delta=delta).view(1, -1)
    log.info(f" x.shape: {x.shape}")

    support = (tr.arange(n_samples) - (n_samples // 2)) / sr
    x2 = ChirpletSynth.generate_am_chirp(support, f0_hz, am_hz, fm_oct_hz, bw_n_samples, delta).view(1, -1)
    log.info(f"x2.shape: {x2.shape}")
    # exit()

    # omega = 4.0
    # am = 1.0
    # f0_hz = 128
    # fm1_hz = 2048
    # n_partials = 8
    # theta = tr.tensor([omega, am, f0_hz, fm1_hz])
    # x = ripple(theta, dur_sec, n_partials, sr)
    # log.info(f"x.shape: {x.shape}")

    J_cqt = 5
    Q_cqt = 12
    hop_len = 256
    eps_cqt = 1e-3
    cqt_params = {
        "sr": sr,
        "bins_per_octave": Q_cqt,
        "n_bins": J_cqt * Q_cqt,
        "hop_length": hop_len,
        "fmin": (0.4 * sr) / (2**J_cqt),
        "output_format": "Magnitude",
        "verbose": False,
    }
    cqt = CQT(**cqt_params)
    y_coords = cqt.frequencies

    # U = cqt(x)
    # U = tr.log1p(U / eps_cqt)
    # log.info(f"U.shape: {U.shape}")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # plot_scalogram(ax, U[0], sr, y_coords=y_coords, hop_len=hop_len, title="Chirp 1")
    # plt.show()

    U2 = cqt(x2)
    U2 = tr.log1p(U2 / eps_cqt)
    log.info(f"U2.shape: {U2.shape}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_scalogram(ax, U2[0], sr, y_coords=y_coords, hop_len=hop_len, title="Chirp 2")
    plt.show()
    plt.close()
