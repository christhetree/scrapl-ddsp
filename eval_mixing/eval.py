import librosa.display
import matplotlib.pyplot as plt
import torchaudio

plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100

from automix.evaluation.utils_evaluation import get_features


if __name__ == "__main__":
    # !wget https://huggingface.co/csteinmetz1/automix-toolkit/resolve/main/drums-test-rock.zip
    # !unzip -o drums-test-rock.zip

    mix_target_path = (
        "drums-test-rock/mix/dry_mix_066_phrase_rock_complex_fast_sticks.wav"
    )
    mix_auto_path_wun = "drums-test-rock/mix/dry_mix_066_phrase_rock_complex_fast_sticks_MixWaveUNet.wav"
    mix_auto_path_dmc = (
        "drums-test-rock/mix/dry_mix_066_phrase_rock_complex_fast_sticks_DMC.wav"
    )

    # Global Settings
    SR = 44100
    max_samples = 262144
    start_sample = 0 * SR
    end_sample = start_sample + max_samples

    # Load the mixes
    fig, axs = plt.subplots(2, 1)

    target_audio, sr = torchaudio.load(mix_target_path)
    target_audio = target_audio[:, start_sample:end_sample]

    librosa.display.waveshow(
        target_audio[0, :].numpy(),
        axis="time",
        sr=SR,
        zorder=3,
        label="human-made",
        color="k",
        ax=axs[0],
    )

    wun_audio, sr = torchaudio.load(mix_auto_path_wun)
    wun_audio = wun_audio[:, start_sample:end_sample]
    librosa.display.waveshow(
        wun_audio[0, :].view(-1).numpy(),
        axis="time",
        sr=SR,
        zorder=3,
        label="MixWaveUNet",
        color="tab:blue",
        ax=axs[0],
        alpha=0.7,
    )
    axs[0].grid(c="lightgray")
    axs[0].legend()

    librosa.display.waveshow(
        target_audio[0, :].numpy(),
        axis="time",
        sr=SR,
        zorder=3,
        label="human-made",
        color="k",
        ax=axs[1],
    )

    dmc_audio, sr = torchaudio.load(mix_auto_path_dmc)
    dmc_audio = dmc_audio[:, start_sample:end_sample]
    librosa.display.waveshow(
        dmc_audio[0, :].view(-1).numpy(),
        axis="time",
        sr=SR,
        zorder=3,
        label="DMC",
        color="tab:orange",
        ax=axs[1],
        alpha=0.7,
    )
    axs[1].grid(c="lightgray")
    axs[1].legend()

    # Compute the loudness, spectral, panning and dynamic features
    target_audio = target_audio.numpy()
    wun_audio = wun_audio.numpy()
    dmc_audio = dmc_audio.numpy()

    wun_features = get_features(target_audio, wun_audio)
    dmc_features = get_features(target_audio, dmc_audio)

    wun_features_mean = {
        k.split("_")[-1]: wun_features.pop(k)
        for k in list(wun_features.keys())
        if k.startswith("mean_mape")
    }
    dmc_features_mean = {
        k.split("_")[-1]: dmc_features.pop(k)
        for k in list(dmc_features.keys())
        if k.startswith("mean_mape")
    }

    # Plots averages features
    plt.bar(
        *zip(*wun_features_mean.items()),
        alpha=0.5,
        fill=True,
        color="tab:blue",
        label="MixWaveUNet"
    )
    plt.bar(
        *zip(*dmc_features_mean.items()),
        alpha=0.5,
        fill=True,
        color="tab:orange",
        label="DMC"
    )
    plt.xticks(rotation=-90)
    plt.ylabel("MAPE")
    plt.legend()
    plt.show()

    # Plots all features
    plt.bar(
        *zip(*wun_features.items()),
        alpha=0.5,
        fill=True,
        color="tab:blue",
        label="MixWaveUNet"
    )
    plt.bar(
        *zip(*dmc_features.items()),
        alpha=0.5,
        fill=True,
        color="tab:orange",
        label="DMC"
    )
    plt.axvline(1.5, 0, 1, linestyle="--", alpha=0.5, color="k", linewidth=0.75)
    plt.axvline(6.5, 0, 1, linestyle="--", alpha=0.5, color="k", linewidth=0.75)
    plt.axvline(10.5, 0, 1, linestyle="--", alpha=0.5, color="k", linewidth=0.75)
    plt.xticks(rotation=-90)
    plt.ylabel("MAPE")
    plt.legend()
    plt.show()
