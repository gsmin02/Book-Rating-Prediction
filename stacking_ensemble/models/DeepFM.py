import torch
import torch.nn as nn
from _helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base

class DeepFM(nn.Module):
    def __init__(self, model_kwargs, field_dims):
        super().__init__()
        self.field_dims = field_dims
        factor_dim = model_kwargs.get('embed_dim', 16)
        mlp_dims = model_kwargs.get('mlp_dims', [16, 32])
        batchnorm = model_kwargs.get('batchnorm', True)
        dropout = model_kwargs.get('dropout', 0.2)

        # First-order linear term
        self.linear = FeaturesLinear(self.field_dims)

        # Sparse feature embedding
        self.embedding = FeaturesEmbedding(self.field_dims, factor_dim)

        # FM second-order interaction
        self.fm = FMLayer_Dense()

        # Deep network
        self.dnn = MLP_Base(
            input_dim=factor_dim * len(self.field_dims),
            embed_dims=mlp_dims,
            batchnorm=batchnorm,
            dropout=dropout,
            output_layer=True
        )

    def forward(self, x: torch.Tensor):
        # First-order term
        first_order = self.linear(x).squeeze(1)

        # Sparse to dense embeddings
        embedding = self.embedding(x)  # (batch_size, num_fields, embed_dim)

        # FM second-order term
        second_order = self.fm(embedding)

        # Deep network
        deep_out = self.dnn(embedding.view(-1, embedding.size(1) * embedding.size(2))).squeeze(1)

        return first_order + second_order + deep_out