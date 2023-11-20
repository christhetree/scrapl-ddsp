import numpy as np
import torch


def gaussian(M, std, sym=True):
    """Gaussian window converted from scipy.signal.gaussian"""
    if M < 1:
        return torch.array([])
    if M == 1:
        return torch.ones(1, "d")
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0

    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def make_hann_window(n_samples: int) -> torch.Tensor:
    x = torch.arange(n_samples)
    y = torch.sin(torch.pi * x / n_samples) ** 2
    return y


def get_amplitudes(theta_density, n_events):
    offset = 0.25 * theta_density + 0.75 * theta_density**2
    event_ids = torch.tensor(np.arange(n_events)).type(torch.float32)
    sigmoid_operand = (1 - theta_density) * n_events * (event_ids / n_events - offset)
    amplitudes = 1 - torch.sigmoid(2 * sigmoid_operand)
    amplitudes = amplitudes / torch.max(amplitudes)
    return amplitudes


def get_slope(theta_slope, Q, hop_length, sr):
    """theta_slope --> ±1 correspond to a near-vertical line.
    theta_slope = 0 corresponds to a horizontal line.
    The output is measured in octaves per second."""
    typical_slope = sr / (Q * hop_length)
    return torch.tan(torch.tensor(theta_slope * np.pi / 2)) * typical_slope / 4


def generate_chirp_texture(
    theta_density,
    theta_slope,
    *,
    duration,
    event_duration,
    sr,
    fmin,
    fmax,
    n_events,
    Q,
    hop_length,
    seed,
):
    # Set random seed
    random_state = np.random.RandomState(seed)

    # Define constant log(2)
    const_log2 = torch.log(torch.tensor(2.0))

    # Define the amplitudes of the events
    amplitudes = get_amplitudes(theta_density, n_events)

    # Define the slope of the events
    gamma = get_slope(theta_slope, Q, hop_length, sr)
    print(f"gamma: {gamma}")

    # Draw onsets at random
    chirp_duration = event_duration
    chirp_length = torch.tensor(chirp_duration * sr).int()
    rand_onsets = torch.from_numpy(random_state.rand(n_events))
    # TODO
    onsets = rand_onsets * (duration * sr / 2 - chirp_length) + duration * sr / 4
    onsets = torch.floor(onsets).type(torch.int64)

    # Draw frequencies at random
    log2_fmin = torch.log2(torch.tensor(fmin))
    log2_fmax = torch.log2(torch.tensor(fmax))
    rand_pitches = torch.from_numpy(random_state.rand(n_events))
    log2_frequencies = log2_fmin + rand_pitches * (log2_fmax - log2_fmin)
    frequencies = torch.pow(2.0, log2_frequencies)

    # Generate events one by one
    X = torch.zeros(duration * sr, n_events)
    time = torch.arange(chirp_length) / sr - chirp_duration / 2
    event_ids = torch.arange(n_events)
    patch_zip = zip(event_ids, onsets, amplitudes, frequencies)

    # Define envelope
    # sigma_window = 0.1
    # envelope = gaussian(chirp_length, sigma_window)
    envelope = make_hann_window(chirp_length)
    # envelope = 1.0

    for event_id, onset, amplitude, frequency in patch_zip:
        phase = torch.expm1(gamma * const_log2 * time) / (gamma * const_log2)
        phase[torch.isnan(phase)] = time[torch.isnan(phase)]
        chirp = torch.sin(2 * torch.pi * frequency * phase)
        offset = onset + chirp_length
        X[onset:offset, event_id] = chirp * amplitude * envelope * torch.sqrt(frequency)

    # Mix events
    x = X.sum(axis=-1)

    # Renormalize
    x = x / torch.norm(x)

    return x
