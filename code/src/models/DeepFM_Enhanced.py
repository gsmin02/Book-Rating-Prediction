"""
Enhanced DeepFM Model
- 전략 2: Sparse(범주형) + Dense(수치형) 피처 함께 사용
- 전략 3: Cold Start 인식 및 처리

기존 DeepFM 대비 변경점:
1. Dense 피처를 Deep Network에 직접 연결
2. FM layer에서도 Dense 피처 활용 옵션
3. Global/User/Item/Author bias 명시적 모델링
"""

import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    """Sparse 피처의 1차 상호작용 (Linear)"""
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    """Sparse 피처를 Dense 벡터로 임베딩"""
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FMLayer_Dense(nn.Module):
    """FM의 2차 상호작용 계산"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (batch_size, num_fields, embed_dim)
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)


class MLP_Base(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, input_dim, embed_dims, batchnorm=True, dropout=0.2, output_layer=True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


import numpy as np

class DeepFM_Enhanced(nn.Module):
    """
    Enhanced DeepFM: Sparse + Dense 피처 통합 모델
    
    구조:
    1. First-order: Linear(sparse) + Dense 피처
    2. Second-order: FM(sparse embeddings)
    3. Deep: MLP(concat(sparse embeddings, dense features))
    
    Args:
        args: 모델 하이퍼파라미터
            - embed_dim: Sparse 피처 임베딩 차원
            - mlp_dims: MLP 히든 레이어 차원 리스트
            - batchnorm: 배치 정규화 사용 여부
            - dropout: 드롭아웃 비율
        data: 데이터 정보
            - field_dims: 각 sparse 피처의 고유값 수
            - dense_dim: Dense 피처 수
    """
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.dense_dim = data.get('dense_dim', 0)
        self.embed_dim = args.embed_dim
        self.num_fields = len(self.field_dims)
        
        # ============ Sparse 피처 처리 ============
        # 1차 상호작용 (Linear)
        self.linear = FeaturesLinear(self.field_dims)
        
        # Embedding
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        # FM Layer (2차 상호작용)
        self.fm = FMLayer_Dense()
        
        # ============ Dense 피처 처리 ============
        if self.dense_dim > 0:
            # Dense 피처의 1차 기여
            self.dense_linear = nn.Linear(self.dense_dim, 1)
            
            # Dense 피처를 임베딩 차원으로 변환 (FM에 참여시킬 경우)
            # self.dense_embedding = nn.Linear(self.dense_dim, args.embed_dim)
        
        # ============ Deep Network ============
        # 입력: (sparse embedding flatten) + (dense features)
        deep_input_dim = (args.embed_dim * self.num_fields) + self.dense_dim
        
        self.dnn = MLP_Base(
            input_dim=deep_input_dim,
            embed_dims=args.mlp_dims,
            batchnorm=args.batchnorm,
            dropout=args.dropout,
            output_layer=True
        )
        
        # ============ Global Bias ============
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse, x_dense=None):
        """
        Args:
            x_sparse: (batch_size, num_sparse_fields) - 범주형 피처 인덱스
            x_dense: (batch_size, dense_dim) - 수치형 피처 값
        
        Returns:
            output: (batch_size,) - 예측 rating
        """
        # ============ First-order (Linear) ============
        # Sparse 피처의 1차 기여
        first_order = self.linear(x_sparse).squeeze(1)
        
        # Dense 피처의 1차 기여
        if self.dense_dim > 0 and x_dense is not None:
            first_order = first_order + self.dense_linear(x_dense).squeeze(1)
        
        # ============ Second-order (FM) ============
        # Sparse 피처 임베딩
        sparse_embed = self.embedding(x_sparse)  # (batch, num_fields, embed_dim)
        
        # FM 계산
        second_order = self.fm(sparse_embed)
        
        # ============ Deep Network ============
        # Sparse embedding flatten
        sparse_flat = sparse_embed.view(-1, self.num_fields * self.embed_dim)
        
        # Dense 피처와 concat
        if self.dense_dim > 0 and x_dense is not None:
            deep_input = torch.cat([sparse_flat, x_dense], dim=1)
        else:
            deep_input = sparse_flat
        
        deep_out = self.dnn(deep_input).squeeze(1)
        
        # ============ 최종 출력 ============
        output = self.global_bias + first_order + second_order + deep_out
        
        return output


class DeepFM_Enhanced_WithBias(nn.Module):
    """
    DeepFM Enhanced + 명시적 Bias Term
    
    EDA 발견: User bias가 매우 큼 (까다로운 User 3.51 vs 후한 User 8.04)
    → User/Item/Author bias를 명시적으로 모델링
    
    예측 = global_bias + user_bias + item_bias + author_bias + FM + Deep
    """
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.dense_dim = data.get('dense_dim', 0)
        self.embed_dim = args.embed_dim
        self.num_fields = len(self.field_dims)
        
        # user_id와 isbn의 field_dim (첫 번째, 두 번째로 가정)
        self.num_users = self.field_dims[0]
        self.num_items = self.field_dims[1]
        
        # ============ Explicit Bias Terms ============
        self.global_bias = nn.Parameter(torch.tensor([data['stats']['global_mean']]))
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        
        # Bias 초기화 (0으로)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        # ============ Sparse 피처 처리 ============
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.fm = FMLayer_Dense()
        
        # ============ Dense 피처 처리 ============
        if self.dense_dim > 0:
            self.dense_linear = nn.Linear(self.dense_dim, 1)
        
        # ============ Deep Network ============
        deep_input_dim = (args.embed_dim * self.num_fields) + self.dense_dim
        
        self.dnn = MLP_Base(
            input_dim=deep_input_dim,
            embed_dims=args.mlp_dims,
            batchnorm=args.batchnorm,
            dropout=args.dropout,
            output_layer=True
        )

    def forward(self, x_sparse, x_dense=None):
        # User, Item 인덱스 추출
        user_idx = x_sparse[:, 0]
        item_idx = x_sparse[:, 1]
        
        # ============ Bias Terms ============
        bias = (
            self.global_bias 
            + self.user_bias(user_idx).squeeze(1) 
            + self.item_bias(item_idx).squeeze(1)
        )
        
        # ============ FM ============
        sparse_embed = self.embedding(x_sparse)
        fm_out = self.fm(sparse_embed)
        
        # ============ Dense Linear ============
        if self.dense_dim > 0 and x_dense is not None:
            dense_out = self.dense_linear(x_dense).squeeze(1)
        else:
            dense_out = 0
        
        # ============ Deep Network ============
        sparse_flat = sparse_embed.view(-1, self.num_fields * self.embed_dim)
        if self.dense_dim > 0 and x_dense is not None:
            deep_input = torch.cat([sparse_flat, x_dense], dim=1)
        else:
            deep_input = sparse_flat
        deep_out = self.dnn(deep_input).squeeze(1)
        
        # ============ 최종 출력 ============
        output = bias + fm_out + dense_out + deep_out
        
        return output
