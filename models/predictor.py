import torch
import torch.nn as nn


class EncoderPredictor(nn.Module):
	"""Predictor on top of an already-built (typically frozen) encoder."""

	def __init__(
		self,
		encoder,
		atomic_embedder=None,
	):
		super().__init__()

		if encoder is None:
			raise ValueError("encoder must be provided")
		if not hasattr(encoder, "index_matrix"):
			raise ValueError("encoder must expose index_matrix to infer clique latent shape")

		self.encoder = encoder
		self.atomic_embedder = atomic_embedder

		enc_n_cliques = int(self.encoder.index_matrix.shape[0])
		enc_clique_dim = int(self.encoder.index_matrix.shape[1])
		self.n_cliques = enc_n_cliques
		self.clique_dim = enc_clique_dim
		self.latent_dim = self.n_cliques * self.clique_dim

		# Hardcoded baseline head: flatten the clique latent and map it to one scalar.
		self.regressor = nn.Sequential(
			nn.Linear(self.latent_dim, 128),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(128, 1),
		)

		for param in self.encoder.parameters():
			param.requires_grad = False
		self.encoder.eval()

	# Embed atoms pre-encoding
	def _prepare_atomic(self, atomic):
		# CliqueFlowmerEncoder expects atomic embeddings; allow raw species ids when embedder is provided.
		if atomic.dim() == 2:
			if self.atomic_embedder is None:
				raise ValueError(
					"Received raw atomic ids [B, N] but no atomic_embedder was provided. "
					"Pass model.atomic_emb when constructing EncoderPredictor."
				)
			return self.atomic_embedder(atomic.long())

		if atomic.dim() != 3:
			raise ValueError("atomic must have shape [B, N] (ids) or [B, N, D] (embeddings)")

		return atomic

	# Run encoded inputs through regression head
	def predict_from_latent(self, z):
		"""Predict from precomputed clique-structured latent z of shape [B, C, D]."""
		if z.dim() != 3:
			raise ValueError("Expected latent tensor with shape [batch, n_cliques, clique_dim]")

		if z.shape[1] != self.n_cliques or z.shape[2] != self.clique_dim:
			raise ValueError(
				f"Expected latent shape [B, {self.n_cliques}, {self.clique_dim}], "
				f"got {tuple(z.shape)}"
			)

		z = z.reshape(z.shape[0], -1)
		return self.regressor(z).squeeze(-1)

	# Run inputs through encoder
	def encode(self, abc, angles, atomic, pos, mask):
		"""Encode inputs into clique latents shaped [B, n_cliques, clique_dim]."""
		atomic = self._prepare_atomic(atomic)

		with torch.no_grad():
			mu, _ = self.encoder(abc, angles, atomic, pos, mask, separate=True)

		return mu

	def forward(self, abc, angles, atomic, pos, mask):
		z = self.encode(abc, angles, atomic, pos, mask)
		return self.predict_from_latent(z)
