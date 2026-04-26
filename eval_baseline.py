"""Evaluate the HuggingFace CliqueFlowmer target_regressor on test_form.pkl.

Loads CliqueFlowmer-MP20-Eform.pth, encodes test structures through the
pretrained encoder, and runs predictions through model.target_regressor
(the stable Polyak-averaged head). Prints the same metrics as the
evaluate() function in train_predictor.py.
"""

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
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_state_dict(obj):
    """Extract state dict from checkpoint objects that wrap it."""
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj


def load_hf_model():
    """Download CliqueFlowmer-MP20-Eform from HuggingFace and return the model."""
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CHECKPOINT)

    from configs.mp20.cliqueflowmer import get_config
    config = get_config()
    model_kwargs = dict(config.model)
    model_cls = model_kwargs.pop("cls")
    model_kwargs.pop("mle_prior", None)

    model = getattr(models, model_cls)(**model_kwargs).to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _resolve_state_dict(raw)
    # Strip DataParallel "module." prefix if present
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # index_matrix is a plain tensor attribute (not a buffer), so .to(device) misses it
    model.encoder.index_matrix = model.encoder.index_matrix.to(device)
    model.index_matrix = model.index_matrix.to(device)

    return model


def evaluate_baseline(model, test_data):
    """Run evaluation and print metrics matching train_predictor.py's evaluate()."""
    test_structures = test_data["structures"]
    test_targets = test_data["targets"]
    target_std = float(test_data.get("target_std", 1.0))

    test_dataset = tools.MatbenchDataset(test_structures, test_targets, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tools.collate_structure,
    )

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for abc, angles, species, positions, mask, batch_targets in test_loader:
            abc = abc.to(device)
            angles = angles.to(device)
            species = species.to(device)
            positions = positions.to(device)
            mask = mask.to(device)
            batch_targets = batch_targets.to(device).float().view(-1)

            atomic_emb = model.atomic_emb(species.long())
            z_sep, _ = model.encoder(abc, angles, atomic_emb, positions, mask, separate=True)
            predictions = model.target_regressor(z_sep).view(-1)

            all_preds.append(predictions.cpu())
            all_targets.append(batch_targets.cpu())

    criterion = nn.MSELoss()
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

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
        f"Baseline (HF target_regressor) - "
        f"test_mse: {test_mse:.6f} ({test_mse_phys:.6f} eV²/atom²) - "
        f"test_mae: {test_mae:.6f} ({test_mae_phys:.6f} eV/atom) - "
        f"test_r2: {test_r2:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    model = load_hf_model()
    test_data = loading.load_pickled_object_from_local(TEST_DATA_PATH)
    if test_data is None:
        raise FileNotFoundError(f"Could not find {TEST_DATA_PATH}")
    evaluate_baseline(model, test_data)
