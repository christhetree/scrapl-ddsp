import logging
import os
import re
from string import punctuation

import numpy as np
import torch
import yaml
from g2p_en import G2p

from FastSpeech2.model import FastSpeech2
from FastSpeech2.text import text_to_sequence
from FastSpeech2.utils.tools import to_device
from experiments.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


# def preprocess_mandarin(text, preprocess_config):
#     lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
#
#     phones = []
#     pinyins = [
#         p[0]
#         for p in pinyin(
#             text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
#         )
#     ]
#     for p in pinyins:
#         if p in lexicon:
#             phones += lexicon[p]
#         else:
#             phones.append("sp")
#
#     phones = "{" + " ".join(phones) + "}"
#     print("Raw Text Sequence: {}".format(text))
#     print("Phoneme Sequence: {}".format(phones))
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )
#
#     return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                # mel_lens=torch.tensor([128]).long(),
                # max_mel_len=128,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            mel_predictions = output[1]
            log.info(f"mel_predictions.shape: {mel_predictions.shape}")

            # synth_samples(
            #     batch,
            #     output,
            #     vocoder,
            #     model_config,
            #     preprocess_config,
            #     train_config["path"]["result_path"],
            # )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_step", type=int, required=True)
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     choices=["batch", "single"],
    #     required=True,
    #     help="Synthesize a whole dataset or a single sentence",
    # )
    # parser.add_argument(
    #     "--source",
    #     type=str,
    #     default=None,
    #     help="path to a source file with format like train.txt and val.txt, for batch mode only",
    # )
    # parser.add_argument(
    #     "--text",
    #     type=str,
    #     default=None,
    #     help="raw text to synthesize, for single-sentence mode only",
    # )
    # parser.add_argument(
    #     "--speaker_id",
    #     type=int,
    #     default=0,
    #     help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    # )
    # parser.add_argument(
    #     "-p",
    #     "--preprocess_config",
    #     type=str,
    #     required=True,
    #     help="path to preprocess.yaml",
    # )
    # parser.add_argument(
    #     "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    # )
    # parser.add_argument(
    #     "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    # )
    # parser.add_argument(
    #     "--pitch_control",
    #     type=float,
    #     default=1.0,
    #     help="control the pitch of the whole utterance, larger value for higher pitch",
    # )
    # parser.add_argument(
    #     "--energy_control",
    #     type=float,
    #     default=1.0,
    #     help="control the energy of the whole utterance, larger value for larger volume",
    # )
    # parser.add_argument(
    #     "--duration_control",
    #     type=float,
    #     default=1.0,
    #     help="control the speed of the whole utterance, larger value for slower speaking rate",
    # )
    # args = parser.parse_args()

    # import nltk
    # nltk.download('averaged_perceptron_tagger_eng')

    device = "cpu"
    preprocess_config_path = os.path.join(CONFIGS_DIR, "fastspeech2/preprocess.yaml")
    model_config_path = os.path.join(CONFIGS_DIR, "fastspeech2/model.yaml")
    ckpt_path = None
    mode = "single"
    source = None
    speaker_id = 0
    text = "Pigs are flying!"
    pitch_control = 1.0
    energy_control = 1.0
    duration_control = 1.0
    max_text_len = 100

    # Check source texts
    if mode == "batch":
        assert source is not None and text is None
    if mode == "single":
        assert source is None and text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(preprocess_config_path, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
    train_config = None
    configs = (preprocess_config, model_config, train_config)

    # Get model
    # model = get_model(args, configs, device, train=False)
    model = FastSpeech2(preprocess_config, model_config).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_ = False

    # Load vocoder
    # vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    # if mode == "batch":
    #     # Get dataset
    #     dataset = TextDataset(source, preprocess_config)
    #     batchs = DataLoader(
    #         dataset,
    #         batch_size=8,
    #         collate_fn=dataset.collate_fn,
    #     )

    if mode == "single":
        ids = raw_texts = [text[:max_text_len]]
        speakers = np.array([speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(text, preprocess_config)])
        # elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        #     texts = np.array([preprocess_mandarin(text, preprocess_config)])
        else:
            raise ValueError("Only English is supported.")
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = pitch_control, energy_control, duration_control

    restore_step = None
    vocoder = None
    synthesize(model, restore_step, configs, vocoder, batchs, control_values)
