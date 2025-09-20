import logging
import os
import tempfile

# Prevents a bug with PyTorch and CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import yaml

from experiments.cli import CustomLightningCLI
from experiments.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

torch.set_float32_matmul_precision("high")


def load_yaml_recursive(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def _resolve(node, base_dir):
        if isinstance(node, dict):
            return {k: _resolve(v, base_dir) for k, v in node.items()}
        elif isinstance(node, list):
            return [_resolve(v, base_dir) for v in node]
        elif isinstance(node, str) and node.endswith((".yaml", ".yml")):
            subpath = os.path.join(base_dir, node)
            return load_yaml_recursive(subpath)  # recurse
        else:
            return node

    return _resolve(cfg, os.path.dirname(path))


if __name__ == "__main__":
    # config_name = "train.yml"
    # config_name = "train/am_fm/train.yml"

    # config_name = "train/texture/train_ploss.yml"
    # config_name = "train/texture/train_mss.yml"
    # config_name = "train/texture/train_mss_meso.yml"
    # config_name = "train/texture/train_mss_meso_log.yml"
    # config_name = "train/texture/train_rand_mss.yml"
    # config_name = "train/texture/train_mss_revisited.yml"
    # config_name = "train/texture/train_clap.yml"
    # config_name = "train/texture/train_clap_testing.yml"
    # config_name = "train/texture/train_panns_wglm.yml"
    # config_name = "train/texture/train_jtfst.yml"
    # config_name = "train/texture/train_scrapl_adam.yml"
    # config_name = "train/texture/train_scrapl_saga_adam.yml"
    # config_name = "train/texture/train_scrapl_pwa.yml"
    # config_name = "train/texture/train_scrapl_saga_pwa.yml"
    # config_name = "train/texture/train_scrapl_saga_pwa_warmup.yml"
    # config_name = "train/texture/train_scrapl_saga_pwa__adaptive_n_batches_1_n_iter_20_param_agg_none.yml"   # min = 0.000101, max = 0.020284
    # config_name = "train/texture/train_scrapl_saga_pwa__adaptive_n_batches_1_n_iter_20_param_agg_mean.yml"   # min = 0.000087, max = 0.019774
    # config_name = "train/texture/train_scrapl_saga_pwa__adaptive_n_batches_1_n_iter_20_param_agg_max.yml"    # min = 0.000081, max = 0.020218
    # config_name = "train/texture/train_scrapl_saga_pwa__adaptive_n_batches_10_n_iter_20_param_agg_none.yml"  # min = 0.000249, max = 0.025870

    # config_name = "train/chirplet/train_scrapl_saga_pwa_warmup.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__am_lo_fm_lo.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__adaptive_n_batches_1_n_iter_20_param_agg_none__am_lo_fm_lo.yml"  # min = 0.000003, max = 0.012895
    # config_name = "train/chirplet/train_scrapl_saga_pwa__adaptive_n_batches_10_n_iter_20_param_agg_none__am_lo_fm_lo.yml"  # min = 0.000000, max = 0.014541

    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_lo_fm_lo.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_lo_fm_med.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_hi_fm_med.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_hi_fm_hi.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_lo_fm_lo__probs_n_batches_1.yml"  # min = 0.000002, max = 0.027627
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_lo_fm_med__probs_n_batches_1.yml"  # min = 0.000001, max = 0.020115
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_hi_fm_med__probs_n_batches_1.yml"  # min = 0.000001, max = 0.015880
    # config_name = "train/chirplet/train_scrapl_saga_pwa__2_am_hi_fm_hi__probs_n_batches_1.yml"  # min = 0.000000, max = 0.016704

    # config_name = "train/chirplet/train_scrapl_saga_pwa.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_mean.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__probs__n_batches_20__n_iter_20__min_prob_frac_0.0__param_agg_None.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa_ds_update.yml"

    # config_name = "train/chirplet/train_scrapl_adam.yml"
    # config_name = "train/chirplet/train_scrapl_pwa.yml"
    # config_name = "train/chirplet/train_scrapl_saga.yml"
    # config_name = "train/chirplet/train_scrapl_saga_am_or_fm.yml"
    # config_name = "train/chirplet/train_scrapl_saga_bin.yml"
    # config_name = "train/chirplet/train_scrapl_saga_ds_w0.yml"
    # config_name = "train/chirplet/train_scrapl_saga_d_w0.yml"
    # config_name = "train/chirplet/train_scrapl_saga_s_w0.yml"

    # config_name = "train/chirplet/am/train_scrapl_adam.yml"
    # config_name = "train/chirplet/am/train_scrapl_pwa.yml["
    # config_name = "train/chirplet/am/train_scrapl_saga.yml"
    # config_name = "train/chirplet/am/train_scrapl_saga_am.yml"
    # config_name = "train/chirplet/am/train_scrapl_saga_bin.yml"
    # config_name = "train/chirplet/am/train_scrapl_saga_w0.yml"

    # config_name = "train/chirplet/fm/train_scrapl_adam.yml"
    # config_name = "train/chirplet/fm/train_scrapl_pwa.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga_fm.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga_bin.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga_w0.yml"

    # ==================================================================================
    # config_name = "eval_mixing/train.yml"
    # config_name = "eval_808/train.yml"  # min = 0.000082, max = 0.016156
    # config_name = "eval_808/train_mars_808.yml"  # min = 0.000138, max = 0.024875
    # config_name = "eval_808/train_mss_micro.yml"
    # config_name = "eval_808/train_mss_meso.yml"
    # config_name = "eval_808/train_mss_lin_micro.yml"
    # config_name = "eval_808/train_mss_lin_meso.yml"
    # config_name = "eval_808/train_mss_log_micro.yml"
    # config_name = "eval_808/train_mss_log_meso.yml"
    # config_name = "eval_808/train_mss_rev_micro.yml"
    # config_name = "eval_808/train_mss_rev_meso.yml"
    # config_name = "eval_808/train_rand_mss_micro.yml"
    # config_name = "eval_808/train_rand_mss_meso.yml"
    # config_name = "eval_808/train_scrapl_micro.yml"
    # config_name = "eval_808/train_scrapl_meso.yml"
    # config_name = "eval_808/train_scrapl_micro__probs_n_batches_1.yml"  # min = 0.000058, max = 0.014286
    config_name = "eval_808/train_scrapl_meso__probs_n_batches_1.yml"  # min = 0.000058, max = 0.014280
    # config_name = "eval_808/train_scrapl_no_log1p_micro.yml"
    # config_name = "eval_808/train_scrapl_no_log1p_meso.yml"
    # config_name = "eval_808/train_scrapl_og_micro.yml"
    # config_name = "eval_808/train_scrapl_og_meso.yml"
    # config_name = "eval_808/train_jtfs_micro.yml"
    # config_name = "eval_808/train_jtfs_meso.yml"

    log.info(f"Running with config: {config_name}")
    # seeds = None
    seeds = list(range(40))
    # seeds = list(range(26, 40))

    config_path = os.path.join(CONFIGS_DIR, config_name)

    if seeds is None:
        cli = CustomLightningCLI(
            args=["fit", "-c", config_path],
            trainer_defaults=CustomLightningCLI.make_trainer_defaults()
        )
        trainer = cli.trainer
        trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
    else:
        log.info(f"Running with seeds: {seeds}")
        for seed in seeds:
            log.info(f"Current seed_everything value: {seed}")
            config = load_yaml_recursive(config_path)
            config["data"]["init_args"]["shuffle_seed"] = seed
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
                yaml.dump(config, tmp)
                new_config_path = tmp.name

            cli = CustomLightningCLI(
                args=["fit", "-c", new_config_path, "--seed_everything", str(seed)],
                # run=False, args=["-c", new_config_path, "--seed_everything", str(seed)],
                trainer_defaults=CustomLightningCLI.make_trainer_defaults()
            )
            trainer = cli.trainer
            trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
