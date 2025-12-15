import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base



# DNN과 FM을 결합한 DeepFM 모델을 구현합니다.
class DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim
        self.num_continuous = len(data['continuous_field_names'])
        self.text_embedding_cols = data['text_embedding_cols']
        self.text_embed_dim = args.text_embed_dim
        self.num_text_features = len(self.text_embedding_cols) * self.text_embed_dim

        self.linear_cont = nn.Linear(self.num_continuous, 1)

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()
        
        # deep network를 통해 feature를 학습하는 부분
        self.dnn = MLP_Base(
                             input_dim=(args.embed_dim * len(self.field_dims) + self.num_continuous + self.num_text_features),
                             embed_dims=args.mlp_dims,
                             batchnorm=args.batchnorm,
                             dropout=args.dropout,
                             output_layer=True
                            )


    def forward(self, x: torch.Tensor, x_cont: torch.Tensor, x_text: torch.Tensor):
        # first-order interaction / sparse feature only
        first_order_sparse = self.linear(x).squeeze(1)
    
        # continuous feature의 1차 항 추가
        first_order_cont = self.linear_cont(x_cont).squeeze(1)
        
        first_order = first_order_sparse + first_order_cont

        # sparse to dense
        embedding = self.embedding(x)  # (batch_size, num_fields, embed_dim)

        # second-order interaction / dense
        second_order = self.fm(embedding)

        deep_input_sparse = embedding.view(-1, embedding.size(1) * embedding.size(2))
        x_text_flat = x_text.view(x_text.size(0), -1)

        deep_input = torch.cat([deep_input_sparse, x_cont, x_text_flat], dim=1)
        
        # deep network를 통해 feature를 학습하는 부분
        deep_out = self.dnn(deep_input).squeeze(1)

        final_score = first_order + second_order + deep_out

        output = torch.clamp(final_score, min=1.0, max=10.0)

        return output