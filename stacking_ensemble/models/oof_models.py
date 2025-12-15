import lightgbm as lgb
import torch
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from ncf_wrapper import NCFWrapper
from deepFM_wrapper import DeepFMWrapper

def lgb_oof_model(lr):
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=lr,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        max_depth=-1
    )

def create_ncf_model(field_dims):
    return NCFWrapper(
        field_dims=field_dims,  # 각 feature cardinality
        embed_dim=16,
        mlp_dims=[16, 32],
        batchnorm=True,
        dropout=0.2,
        epochs=10,
        batch_size=1024,
        lr=1e-3
    )

def create_deepfm_model(field_dims):
    return DeepFMWrapper(
        field_dims=field_dims,
        embed_dim=16,
        mlp_dims=[16, 32],
        batchnorm=True,
        dropout=0.2,
        epochs=10,
        batch_size=1024,
        lr=1e-4
    )