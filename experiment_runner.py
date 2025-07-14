#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiment_runner.py  —  triple-mode driver (train / test / all)

• train : iterate EMISSION_SWEEP, launch train_gnn.py & train_llm.py, save models
• test  : for every emission tag with *both* models ready, run test.py once
• all   : per-config pipeline ＝ train-GNN ➔ train-LLM ➔ test
"""

import argparse, json, subprocess, pathlib, shutil, os
from tqdm import tqdm
import re
import datetime

# ------------------------------------------------------------------
# Configurable constants
# ------------------------------------------------------------------
BASE_CFG  = pathlib.Path("config.json")
SAVE_ROOT = pathlib.Path("save")
LOG_ROOT  = pathlib.Path("logs")
MODEL_DIR = pathlib.Path("model")
CONFIG_DIR = pathlib.Path("configs")
CONFIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_ROOT.mkdir(exist_ok=True)

EMISSION_SWEEP = [
    # [4.9,1.0,1.9,2.8,0.3] #0706 lam_all
    # [0.3,1.2,2.1,3.5,4.8] #0706 lam0406
    # [0.2, 3.2, 2.8, 1.6, 0.9] # 0708 lam_all
    # [3.2, 2.0, 2.8, 0.2, 1.1],
    # [0.8, 3.2, 0.2, 1.4, 2.4]

    # 7 group rate2
    # [1.8, 1.2, 1.6, 1, 0.9], # 0711 rate_2
    # [1.3, 0.9, 1.8, 1.4, 1.1], # 0711 rate_2
    # [2.1, 2.6, 2.8, 1.4, 1.8], # 0711 rate_2
    # [2.4, 2.2, 1.8, 1.2, 1.5], # 0711 rate_2
    # [1.5, 1.3, 2.1, 2.6, 1.7], # 0711 rate_2
    # [2.9, 1.5, 1.6, 3, 2.1], # 0711 rate_2
    # [3.3, 3.8, 1.9, 2.2, 2.4], # 0711 rate_2

    #7 group rate4
    # [2.8, 1.2, 3.8, 4.8, 3.3], # 0711 rate_4
    # [2.8, 1.8, 1.2, 4.8, 3.3], # 0711 rate_4
    # [2.8, 4.8, 3.8, 1.2, 3.3], # 0711 rate_4
    # [4.4, 2.5, 3.8, 1.1, 1.8], # 0711 rate_4
    # [1.0, 1.8, 4.0, 2.6, 3.3], # 0711 rate_4
    # [2.8, 3.6, 2.1, 0.9, 1.8], # 0711 rate_4
    # [3.6, 2.0, 0.9, 3.1, 1.5] # 0711 rate_4

    # 6 group rate8
    [5.0, 1.2, 1.6, 1, 0.9], # 0711 rate_8  0.6 4.8
    [4.0, 0.5, 1.8, 3.2, 1.1], # 0711 rate_8 0.5 4
    [0.9, 2.6, 3.2, 1.4, 0.4], # 0711 rate_8 0.4 3.2
    [2.2, 0.6, 1.8, 4.8, 3.5], # 0711 rate_8 0.6 4.8
    [3.2, 4.0, 2.1, 0.5, 1.7], # 0711 rate_8 0.5 4
    [1.0, 0.5, 4.0, 2.6, 3.3], # 0711 rate_8 0.5 4
    # [1.5, 2.8, 3.6, 4.0, 0.5]

    # 9 group rate32
    [3.2, 1.2, 3.8, 0.1, 3.3], # 0711 rate_32 0.1 3.2
    [2.8, 3.2, 1.1, 2.1, 0.1], # 0711 rate_32 0.1 3.2
    [4.9, 0.8, 0.15, 1.8, 3.3], # 0711 rate_32 0.15 4.9
    [2.7, 4.8, 3.8, 0.15, 1.8], # 0711 rate_32 0.15 4.8
    [0.15, 1.8, 4.8, 2.6, 3.3], # 0711 rate_32 0.15 4.8
    [2.9, 0.15, 3.8, 4.8, 1.8], # 0711 rate_32 0.15 4.8
    [0.15, 1.8, 0.15, 2.6, 4.8], # 0711 rate_32 0.15 4.8
    [3.5, 4.8, 3.8, 0.15, 1.8], # 0711 rate_32 0.15 4.8
    [4.8, 1.8, 0.8, 2.6, 0.15], # 0711 rate_32 0.15 4.8



]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def copy_latest_test_results(to_dir, label="gnn"):
    test_root = pathlib.Path("save")
    test_dirs = sorted([d for d in test_root.glob("test_*") if d.is_dir()], reverse=True)
    if not test_dirs:
        print("❌ No test_* directory found to copy.")
        return
    latest = test_dirs[0]
    timestamp = re.findall(r"\d{8}_\d{6}", latest.name)
    timestamp = timestamp[0] if timestamp else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = to_dir / f"test_{label}_{timestamp}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(latest, target)
    print(f"📁 Copied test results to {target}")

def make_cfg(rate_list, lam, tag=None):
    with BASE_CFG.open() as f:
        cfg = json.load(f)
    cfg["train_paras"]["lambda_penalty"] = lam
    cfg["env_paras"]["machine_emission_rates"] = rate_list
    tag = tag or "_".join(f"{r:.1f}" for r in rate_list)
    lam_str = str(lam).replace(".", "_")
    cfg_path = CONFIG_DIR / f"config_em_{tag}_lambda_{lam_str}.json"
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path, tag, lam_str

def launch(cmd, log_path, desc=""):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⏳ {desc}…")
    return subprocess.Popen(
        cmd,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )

def copy_models_for_test(cfg_path, model_path):
    shutil.copy(cfg_path, "config.json")  # set project root config
    for f in MODEL_DIR.glob("*.pt"):
        f.unlink()
    shutil.copy(model_path, MODEL_DIR / model_path.name)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(mode: str, lam: float):
    assert mode in {"train", "test", "all"}, "mode must be train / test / all"

    desc_txt = {"train": "💠  Training",
                "test":  "🧪  Testing",
                "all":   "🔄  Train→Test"}

    for rate_list in tqdm(EMISSION_SWEEP, desc=desc_txt[mode], ncols=100):
        cfg_path, tag, lam_str = make_cfg(rate_list, lam)
        tag_dir = SAVE_ROOT / f"em_{tag}_lambda_{lam_str}"
        tag_dir.mkdir(parents=True, exist_ok=True)
        gnn_model = tag_dir / "gnn.pt"
        llm_model = tag_dir / "llm.pt"

        log_dir = LOG_ROOT / f"em_{tag}_lambda_{lam_str}"
        log_dir.mkdir(parents=True, exist_ok=True)
        runner_log = log_dir / f"runner__em_{tag}__lambda_{lam_str}__{mode}.log"
        runner_log_file = open(runner_log, "a")

        def log(msg):
            print(msg)
            print(msg, file=runner_log_file, flush=True)

        # ------------------------------ TRAIN ------------------------------
        if mode in {"train", "all"}:
            shutil.copy(cfg_path, "config.json")
            p1 = launch(["python", "train_gnn.py", "--config", str(cfg_path)],
                        log_dir / "train_gnn.log", desc=f"[{tag}] GNN training")
            p2 = launch(["python", "train_llm.py"],
                        log_dir / "train_llm.log", desc=f"[{tag}] LLM training")
            p1.wait(); p2.wait()
            log(f"✅ [{tag}] training finished")

        # ------------------------------ TEST ------------------------------
        if mode in {"test", "all"}:
            if not (gnn_model.exists() and llm_model.exists()):
                log(f"⏩ [{tag}] models not ready, skip test.")
                continue

            log(f"🧪 [{tag}] Testing GNN model...")
            copy_models_for_test(cfg_path, gnn_model)
            test_log_gnn = log_dir / "test_gnn.log"
            try:
                subprocess.run(["python", "test.py"],
                               stdout=test_log_gnn.open("w"),
                               stderr=subprocess.STDOUT,
                               check=True)
                log(f"🎯 [{tag}] GNN test completed")
                copy_latest_test_results(to_dir=tag_dir, label="gnn")
            except subprocess.CalledProcessError:
                log(f"💥 [{tag}] GNN test failed, see {test_log_gnn}")

            log(f"🧪 [{tag}] Testing LLM model...")
            copy_models_for_test(cfg_path, llm_model)
            test_log_llm = log_dir / "test_llm.log"
            try:
                subprocess.run(["python", "test.py"],
                               stdout=test_log_llm.open("w"),
                               stderr=subprocess.STDOUT,
                               check=True)
                log(f"🎯 [{tag}] LLM test completed")
                copy_latest_test_results(to_dir=tag_dir, label="llm")
            except subprocess.CalledProcessError:
                log(f"💥 [{tag}] LLM test failed, see {test_log_llm}")

        runner_log_file.close()

# ------------------------------------------------------------------
# Entry point with lambda sweep
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "all"], default="train")
    parser.add_argument("--lambdas", type=float, nargs='+', required=True,
                        help="List of lambda values (space separated)")
    args = parser.parse_args()

    for lam in args.lambdas:
        print(f"🔁 Running for lambda = {lam}")
        main(args.mode, lam)