# CliqueFlowmer — Solving Offline Materials Optimization with CliqueFlowmer

This repository contains the reference implementation for *Solving Offline Materials Optimization with CliqueFlowmer* :contentReference[oaicite:0]{index=0}.

CliqueFlowmer is a domain-specific offline model-based optimization (MBO) method for computational materials discovery. It:

> **encodes** periodic crystal structures into a fixed-dimensional latent vector
>
> **optimizes** that latent with evolution strategies
>
> **decodes** back into a material by (1) autoregressively decoding atom types (beam search) and (2) sampling geometry with a conditional flow model


---

## Repository layout

- `models/`
  - `cliquelowmer.py` — main CliqueFlowmer model (encoder + predictor + decoders)
  - `transformer.py`, `flow.py` — transformer backbones + flow decoder
- `architectures/` — blocks/ops used across models
- `optimization/`
  - `learner.py` — ES / gradient descent learners
  - `design.py` — latent design loop
  - `sun.py` — S.U.N. (stable/unique/novel) utilities
- `data/`
  - `tools.py` — preprocessing, dataset wrappers, structure utils, GCS I/O
  - `constants.py` — atom symbol tables, etc.
- `configs/`
  - `mp20/cliqueflowmer.py` — default config (model + learner + storage)
  - `mp20-bandgap/cliqueflowmer.py` — bandgap variant config
- Top-level scripts
  - `train.py` — distributed training
  - `optimize.py` — latent-space optimization + decode + evaluation
  - `create_m3gnet_eform_targets.py`, `create_megnet_bandgap_targets.py` — target/oracle preparation
  - `sun_from_pickle.py` — S.U.N. evaluation from saved outputs
  - `commands.txt` — minimal launch commands

---

## Setup

### Create an environment

This repo ships a pinned `requirements.txt`. In practice you may need to adjust versions for your system/CUDA and to resolve duplicates (e.g., `Pillow` appears twice) and any unavailable pins.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Training

To train the CliqueFlowmer model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=12346 \
  train.py
```

## Material Discovery

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python optimize.py
```