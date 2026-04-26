import re
import matplotlib.pyplot as plt

# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_formation(path):
    """Parse formation.out (raw data, values already in eV units)."""
    epochs, train_mse, val_mse, val_mae = [], [], [], []
    pat = re.compile(
        r"Epoch (\d+)/\d+ - train_mse: ([\d.]+) - val_mse: ([\d.]+) - val_mae: ([\d.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_mse.append(float(m.group(2)))
                val_mse.append(float(m.group(3)))
                val_mae.append(float(m.group(4)))
    return epochs, train_mse, val_mse, val_mae


def parse_slurm(path):
    """Parse slurm output (z-scored model; extracts physical-unit values in parentheses)."""
    epochs, train_mse, val_mse, val_mae = [], [], [], []
    # matches the parenthesised physical values, e.g. (1.017768 eV2/atom2)
    pat = re.compile(
        r"Epoch (\d+)/\d+ - train_mse: [\d.]+ \(([\d.]+)[^)]+\)"
        r" - val_mse: [\d.]+ \(([\d.]+)[^)]+\)"
        r" - val_mae: [\d.]+ \(([\d.]+)[^)]+\)"
    )
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_mse.append(float(m.group(2)))
                val_mse.append(float(m.group(3)))
                val_mae.append(float(m.group(4)))
    return epochs, train_mse, val_mse, val_mae


# ── Plot helpers ─────────────────────────────────────────────────────────────

def plot_mse(epochs, train_mse, val_mse, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_mse, label="Train MSE")
    ax.plot(epochs, val_mse, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mae(epochs, val_mae, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_mae, color="C2", label="Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

epochs_f, tr_mse_f, v_mse_f, v_mae_f = parse_formation("formation.out")
plot_mse(
    epochs_f, tr_mse_f, v_mse_f,
    ylabel="MSE (eV²/atom²)",
    title="Training & Validation MSE, Raw Data",
    out_path="formation_mse.png",
)
plot_mae(
    epochs_f, v_mae_f,
    ylabel="MAE (eV/atom)",
    title="Validation MAE, Raw Data",
    out_path="formation_mae.png",
)

epochs_s, tr_mse_s, v_mse_s, v_mae_s = parse_slurm("slurm-3121706.out")
plot_mse(
    epochs_s, tr_mse_s, v_mse_s,
    ylabel="MSE (eV²/atom²)",
    title="Training & Validation MSE, Z-Scored Data",
    out_path="slurm_mse.png",
)
plot_mae(
    epochs_s, v_mae_s,
    ylabel="MAE (eV/atom)",
    title="Validation MAE, Z-Scored Data",
    out_path="slurm_mae.png",
)
