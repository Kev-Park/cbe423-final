import os
import pickle
import torch


def load_model_state_dict_from_local(file_path, model, map_location=None):
	if not file_path.endswith(".pth"):
		file_path += ".pth"

	if not os.path.exists(file_path):
		return None

	#
	# Load the state_dict from the local file
	#
	if map_location is None:
		map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

	model.load_state_dict(torch.load(file_path, map_location=map_location))

	return model


def load_pickled_object_from_local(file_path):
	if not file_path.endswith(".pickle") and not file_path.endswith(".pkl"):
		file_path += ".pickle"

	if not os.path.exists(file_path):
		return None

	with open(file_path, "rb") as f:
		obj = pickle.load(f)

	return obj
