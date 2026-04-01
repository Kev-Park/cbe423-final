"""Train predictor utilities on top of a pretrained CliqueFlowmer encoder.

This script follows the same loading pattern as optimize/train:
1) Build the model architecture from config.
2) Load model state dict into that architecture.
3) Extract and optionally freeze only the encoder.
"""

import numpy as np
import torch

from absl import app, flags
from ml_collections import config_flags

import saving
import models


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_string(
	"local_checkpoint_path",
	"",
	"Local checkpoint path (.pth) used to load pretrained model weights.",
)

config_flags.DEFINE_config_file(
	"config",
	"configs/mp20/cliqueflowmer.py",
	"File with hyperparameter configurations.",
	lock_config=False,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pretrained_encoder():
	"""Build full model from config, load checkpoint, and return only encoder."""
	kwargs = dict(**FLAGS.config)

	model_kwargs = dict(kwargs["model"])
	model_cls = model_kwargs.pop("cls")

	torch.manual_seed(FLAGS.seed)
	np.random.seed(FLAGS.seed)

	# This script does encoder-only loading; ignore flow-prior config knobs.
	model_kwargs.pop("mle_prior", None)

	# 1) Define architecture first.
	model = getattr(models, model_cls)(**model_kwargs).to(device)

	# 2) Load pretrained weights into that architecture.
	if not FLAGS.local_checkpoint_path:
		raise ValueError("--local_checkpoint_path is required. This script only loads local state dicts.")

	loaded = saving.load_model_state_dict_from_local(
		FLAGS.local_checkpoint_path,
		model,
	)

	if loaded is None:
		raise FileNotFoundError(f"Could not find local checkpoint: {FLAGS.local_checkpoint_path}")

	model = loaded.to(device)
	model.eval()

	# 3) Extract encoder-only module.
	encoder = model.encoder
	for param in encoder.parameters():
		param.requires_grad = False
	encoder.eval()

	return encoder, model


def main(_):
	encoder, _ = build_pretrained_encoder()

	n_params = sum(p.numel() for p in encoder.parameters())
	n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
	print("Loaded pretrained encoder successfully.")
	print(f"Encoder params: {n_params}")
	print(f"Trainable params: {n_trainable}")


if __name__ == "__main__":
	app.run(main)

