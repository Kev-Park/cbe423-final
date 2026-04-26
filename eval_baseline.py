"""Evaluate the HuggingFace CliqueFlowmer regressor heads on test_form.pkl.

Loads CliqueFlowmer-MP20-Eform.pth and reports metrics for model.regressor
(directly-trained head).  A mean-shift calibration bias is estimated on
val.pkl (held-out from test) and subtracted from test predictions so that
the comparison with the locally-trained EncoderPredictor is in-distribution.

Normalization stats are computed from raw_data/train.csv since the local
train.pkl was built before z-scoring was added (stores mean=0, std=1).
Metrics match the format from train_predictor.py's evaluate().
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

import loading
import models
import data.tools as tools
import models.graphops as graphops

HF_REPO = "iamkuba/CliqueFlowmer"
HF_CHECKPOINT = "CliqueFlowmer-MP20-Eform.pth"
TEST_DATA_PATH = "preprocessed_data/test_form.pkl"
VAL_DATA_PATH = "preprocessed_data/val.pkl"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj


def load_hf_model():
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CHECKPOINT)

    from configs.mp20.cliqueflowmer import get_config
    config = get_config()
    model_kwargs = dict(config.model)
    model_cls = model_kwargs.pop("cls")
    model_kwargs.pop("mle_prior", None)

    model = getattr(models, model_cls)(**model_kwargs).to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _resolve_state_dict(raw)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    model.encoder.index_matrix = model.encoder.index_matrix.to(device)
    model.index_matrix = model.index_matrix.to(device)

    return model


def _normalize_targets(raw_targets, target_mean, target_std):
    arr = np.array(raw_targets, dtype=np.float32)
    return ((arr - target_mean) / target_std).tolist()


def run_inference(model, data, target_mean, target_std):
    """Return (preds, targets) tensors in z-scored space for the given split."""
    structures = data["structures"]
    raw_targets = data["targets"]
    targets_normalized = _normalize_targets(raw_targets, target_mean, target_std)

    dataset = tools.MatbenchDataset(structures, targets_normalized, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tools.collate_structure,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for abc, angles, species, positions, mask, batch_targets in loader:
            abc = abc.to(device)
            angles = angles.to(device)
            species = species.to(device)
            positions = positions.to(device)
            mask = mask.to(device)
            batch_targets = batch_targets.to(device).float().view(-1)

            atomic_emb = model.atomic_emb(species.long())
            mu, _ = model.encoder(abc, angles, atomic_emb, positions, mask, separate=False)
            preds = model.predict(mu).view(-1)

            all_preds.append(preds.cpu())
            all_targets.append(batch_targets.cpu())

    return torch.cat(all_preds), torch.cat(all_targets)


def print_metrics(label, preds, targets, target_std):
    criterion = nn.MSELoss()
    test_mse = criterion(preds, targets).item()
    diff = preds - targets
    test_mae = torch.mean(torch.abs(diff)).item()
    ss_res = torch.sum(diff ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    phys_diff = diff * target_std
    test_mae_phys = torch.mean(torch.abs(phys_diff)).item()
    test_mse_phys = torch.mean(phys_diff ** 2).item()
    print(
        f"  [{label}] "
        f"test_mse: {test_mse:.6f} ({test_mse_phys:.6f} eV²/atom²) - "
        f"test_mae: {test_mae:.6f} ({test_mae_phys:.6f} eV/atom) - "
        f"test_r2: {test_r2:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    model = load_hf_model()

    from preprocess_data import compute_train_stats
    target_mean, target_std = compute_train_stats("raw_data/train.csv")
    print(f"Normalization: mean={target_mean:.4f}, std={target_std:.4f}", flush=True)

    # --- Calibration: estimate mean-shift bias on val.pkl (not test data) ---
    val_data = loading.load_pickled_object_from_local(VAL_DATA_PATH)
    if val_data is None:
        raise FileNotFoundError(f"Could not find {VAL_DATA_PATH}")

    val_preds, val_targets = run_inference(model, val_data, target_mean, target_std)
    bias = (val_preds - val_targets).mean().item()
    print(
        f"Val calibration: pred_mean={val_preds.mean():.4f}, "
        f"target_mean={val_targets.mean():.4f}, bias={bias:.4f} ({bias * target_std:.4f} eV/atom)",
        flush=True,
    )

    # --- Test evaluation ---
    test_data = loading.load_pickled_object_from_local(TEST_DATA_PATH)
    if test_data is None:
        raise FileNotFoundError(f"Could not find {TEST_DATA_PATH}")

    test_preds, test_targets = run_inference(model, test_data, target_mean, target_std)
    print(
        f"Test: pred_mean={test_preds.mean():.4f}  "
        f"target_mean={test_targets.mean():.4f}",
        flush=True,
    )

    print_metrics("HF regressor (raw)", test_preds, test_targets, target_std)
    print_metrics("HF regressor (bias-corrected)", test_preds - bias, test_targets, target_std)
