"""Train predictor utilities on top of a pretrained CliqueFlowmer encoder.

This script follows the same loading pattern as optimize/train:
1) Build the model architecture from config.
2) Load model state dict into that architecture.
3) Extract and optionally freeze only the encoder.
"""

import os

import numpy as np
import torch

from absl import app, flags
from ml_collections import config_flags

import data.tools as tools
import saving
import models


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_string(
	"save_encoder_state_to",
	"",
	"Optional local path to save the extracted encoder state dict (.pth).",
)
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

	data_kwargs = dict(kwargs["data"])
	model_kwargs = dict(kwargs["model"])
	storage_kwargs = dict(kwargs["storage"])

	task_cls = data_kwargs.pop("task")
	model_cls = model_kwargs.pop("cls")

	bucket = storage_kwargs.pop("bucket")

	torch.manual_seed(FLAGS.seed)
	np.random.seed(FLAGS.seed)

	# Recreate the same optional prior setup used by train/optimize.
	if model_kwargs.pop("mle_prior"):
		data_dir = os.path.join("CliqueFlowmer", "data", "preprocessed", task_cls)
		train_data = tools.load_pickled_object_from_gcs(bucket, os.path.join(data_dir, "train"))
		train_inputs = list(train_data.values())[0]
		length_mle_vals = tools.length_mle(train_inputs)
		model_kwargs["initial_length_dist"] = tools.normal_lengths_from_mle(length_mle_vals, device)

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

	if FLAGS.save_encoder_state_to:
		out_path = FLAGS.save_encoder_state_to
		if not out_path.endswith(".pth"):
			out_path += ".pth"
		torch.save(encoder.state_dict(), out_path)
		print(f"Saved encoder state dict to {out_path}")

	n_params = sum(p.numel() for p in encoder.parameters())
	n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
	print("Loaded pretrained encoder successfully.")
	print(f"Encoder params: {n_params}")
	print(f"Trainable params: {n_trainable}")


if __name__ == "__main__":
	app.run(main)

