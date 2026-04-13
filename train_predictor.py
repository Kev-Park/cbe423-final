"""Train predictor utilities on top of a pretrained CliqueFlowmer encoder.

This script follows the same loading pattern as optimize/train:
1) Build the model architecture from config.
2) Load model state dict into that architecture.
3) Extract and optionally freeze only the encoder.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from absl import app, flags
from ml_collections import config_flags

import loading
import models
import data.tools as tools


FLAGS = flags.FLAGS
PRETRAINED_CHECKPOINT_PATH = "encoder.pth"

flags.DEFINE_integer("seed", 1, "Random seed.")

# hyperparameters for regressor training
flags.DEFINE_integer("epochs", 50, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 32, "Training batch size.")
flags.DEFINE_float("learning_rate", 1e-3, "Training learning rate.")

config_flags.DEFINE_config_file(
	"config",
	"configs/mp20/cliqueflowmer.py",
	"File with hyperparameter configurations.",
	lock_config=False,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pretrained_encoder():
	"""Build the pretrained model, load weights, and return only the encoder."""
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
	loaded = loading.load_model_state_dict_from_local(
		PRETRAINED_CHECKPOINT_PATH,
		model,
	)

	if loaded is None:
		raise FileNotFoundError(f"Could not find local checkpoint: {PRETRAINED_CHECKPOINT_PATH}")

	model = loaded.to(device)
	model.eval()

	# 3) Extract encoder-only module.
	encoder = model.encoder
	for param in encoder.parameters():
		param.requires_grad = False
	encoder.eval()
	if hasattr(encoder, "index_matrix"):
		encoder.index_matrix = encoder.index_matrix.to(device)
	encoder.atomic_embedder = model.atomic_emb

	return encoder

def train(model, train_data, val_data):
	train_structures = train_data["structures"]
	train_targets = train_data["targets"]
	val_structures = val_data["structures"]
	val_targets = val_data["targets"]

	train_dataset = tools.MatbenchDataset(train_structures, train_targets, augment=False)
	val_dataset = tools.MatbenchDataset(val_structures, val_targets, augment=False)

	train_loader = DataLoader(
		train_dataset,
		batch_size=FLAGS.batch_size,
		shuffle=True,
		collate_fn=tools.collate_structure,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=FLAGS.batch_size,
		shuffle=False,
		collate_fn=tools.collate_structure,
	)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.regressor.parameters(), lr=FLAGS.learning_rate)

	model.to(device)
	if model.atomic_embedder is not None:
		model.atomic_embedder.to(device)

	for epoch in range(FLAGS.epochs):
		model.train()
		model.encoder.eval()
		if model.atomic_embedder is not None:
			model.atomic_embedder.eval()

		running_train_loss = 0.0

		for abc, angles, species, positions, mask, batch_targets in train_loader:
			abc = abc.to(device)
			angles = angles.to(device)
			species = species.to(device)
			positions = positions.to(device)
			mask = mask.to(device)
			batch_targets = batch_targets.to(device).float().view(-1)

			optimizer.zero_grad(set_to_none=True)
			predictions = model(abc, angles, species, positions, mask).view(-1)
			loss = criterion(predictions, batch_targets)
			loss.backward()
			optimizer.step()

			running_train_loss += loss.item() * batch_targets.size(0)

		train_mse = running_train_loss / len(train_dataset)

		model.eval()
		running_val_loss = 0.0
		all_preds = []
		all_targets = []
		with torch.no_grad():
			for abc, angles, species, positions, mask, batch_targets in val_loader:
				abc = abc.to(device)
				angles = angles.to(device)
				species = species.to(device)
				positions = positions.to(device)
				mask = mask.to(device)
				batch_targets = batch_targets.to(device).float().view(-1)

				predictions = model(abc, angles, species, positions, mask).view(-1)
				val_loss = criterion(predictions, batch_targets)
				running_val_loss += val_loss.item() * batch_targets.size(0)

				all_preds.append(predictions.detach().cpu())
				all_targets.append(batch_targets.detach().cpu())

		val_mse = running_val_loss / len(val_dataset)

		preds = torch.cat(all_preds)
		targets = torch.cat(all_targets)
		val_mae = torch.mean(torch.abs(preds - targets)).item()
		ss_res = torch.sum((targets - preds) ** 2).item()
		ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
		val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

		print(
			f"Epoch {epoch + 1}/{FLAGS.epochs} - "
			f"train_mse: {train_mse:.6f} - val_mse: {val_mse:.6f} - "
			f"val_mae: {val_mae:.6f} - val_r2: {val_r2:.4f}"
			, flush=True
		)

	return model

def main(_):
	encoder = build_pretrained_encoder()
	train_data = loading.load_pickled_object_from_local("preprocessed_data/train.pkl")
	if train_data is None:
		raise FileNotFoundError("Could not find local data pickle: preprocessed_data/train.pkl")
	val_data = loading.load_pickled_object_from_local("preprocessed_data/val.pkl")
	if val_data is None:
		raise FileNotFoundError("Could not find local data pickle: preprocessed_data/val.pkl")

	predictor = models.EncoderPredictor(
		encoder,
		atomic_embedder=getattr(encoder, "atomic_embedder", None),
	).to(device)
	predictor.eval()

	trained_predictor = train(predictor, train_data, val_data)

	return trained_predictor


if __name__ == "__main__":
	app.run(main)

