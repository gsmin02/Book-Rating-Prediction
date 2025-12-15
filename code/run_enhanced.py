"""
run_enhanced.py - 완전히 독립적으로 실행 가능한 Enhanced DeepFM 스크립트

사용법:
    python run_enhanced.py

모든 코드가 한 파일에 포함되어 있어 별도 import 없이 실행 가능합니다.
"""

import os
import numpy as np
import pandas as pd
import regex
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime


# ============================================================
# 1. 데이터 처리 함수들
# ============================================================

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환'''
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    '''location 데이터를 country, state, city로 분리'''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]
    res = [i if i not in ['n/a', ''] else np.nan for i in res]
    res.reverse()
    
    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):
            res.pop(i)
    
    return res


def process_enhanced_context_data(users, books, train_df, test_df):
    """EDA 발견사항을 반영한 피처 엔지니어링"""
    users_ = users.copy()
    books_ = books.copy()
    
    # Books 전처리
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else 'unknown')
    books_['language'] = books_['language'].fillna('en')
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)
    books_.loc[books_['year_of_publication'] < 1900, 'year_of_publication'] = 1996
    
    # Users 전처리
    users_['age'] = users_['age'].fillna(34)
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)
    users_['location_list'] = users_['location'].apply(lambda x: split_location(x))
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    
    # 통계 계산 (Train에서만!)
    global_mean = train_df['rating'].mean()
    global_std = train_df['rating'].std()
    
    # User 통계
    user_stats = train_df.groupby('user_id').agg(
        user_rating_mean=('rating', 'mean'),
        user_rating_count=('rating', 'count')
    ).reset_index()
    
    # Item 통계
    item_stats = train_df.groupby('isbn').agg(
        item_rating_mean=('rating', 'mean'),
        item_rating_count=('rating', 'count')
    ).reset_index()
    
    # Author 통계
    train_with_author = train_df.merge(books_[['isbn', 'book_author']], on='isbn', how='left')
    author_stats = train_with_author.groupby('book_author').agg(
        author_rating_mean=('rating', 'mean'),
        author_rating_count=('rating', 'count')
    ).reset_index()
    
    # Publisher 통계
    train_with_publisher = train_df.merge(books_[['isbn', 'publisher']], on='isbn', how='left')
    publisher_stats = train_with_publisher.groupby('publisher').agg(
        publisher_rating_mean=('rating', 'mean'),
    ).reset_index()
    
    # Category 통계
    train_with_category = train_df.merge(books_[['isbn', 'category']], on='isbn', how='left')
    category_stats = train_with_category.groupby('category').agg(
        category_rating_mean=('rating', 'mean'),
    ).reset_index()
    
    # Cold Start 식별용
    train_users = set(train_df['user_id'].unique())
    train_items = set(train_df['isbn'].unique())
    
    # 통계 병합
    users_ = users_.merge(user_stats, on='user_id', how='left')
    books_ = books_.merge(item_stats, on='isbn', how='left')
    books_ = books_.merge(author_stats, on='book_author', how='left')
    books_ = books_.merge(publisher_stats, on='publisher', how='left')
    books_ = books_.merge(category_stats, on='category', how='left')
    
    return users_, books_, {
        'global_mean': global_mean,
        'global_std': global_std,
        'train_users': train_users,
        'train_items': train_items,
        'author_stats': author_stats,
    }


class EnhancedDataset(Dataset):
    """Sparse + Dense 피처 Dataset"""
    def __init__(self, sparse_data, dense_data, rating=None):
        self.sparse_data = sparse_data
        self.dense_data = dense_data
        self.rating = rating
        
    def __len__(self):
        return len(self.sparse_data)
    
    def __getitem__(self, idx):
        result = {
            'sparse': torch.tensor(self.sparse_data[idx], dtype=torch.long),
            'dense': torch.tensor(self.dense_data[idx], dtype=torch.float32),
        }
        if self.rating is not None:
            result['rating'] = torch.tensor(self.rating[idx], dtype=torch.float32)
        return result


# ============================================================
# 2. 모델 정의
# ============================================================

class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FMLayer(nn.Module):
    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class DeepFM_Enhanced(nn.Module):
    def __init__(self, field_dims, dense_dim, embed_dim=16, mlp_dims=[64, 32, 16], dropout=0.3):
        super().__init__()
        self.field_dims = field_dims
        self.dense_dim = dense_dim
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        
        # Sparse 피처 처리
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FMLayer()
        
        # Dense 피처 처리
        if dense_dim > 0:
            self.dense_linear = nn.Linear(dense_dim, 1)
        
        # Deep Network
        deep_input_dim = (embed_dim * self.num_fields) + dense_dim
        self.dnn = MLP(deep_input_dim, mlp_dims, dropout)
        
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse, x_dense=None):
        # First-order
        first_order = self.linear(x_sparse).squeeze(1)
        if self.dense_dim > 0 and x_dense is not None:
            first_order = first_order + self.dense_linear(x_dense).squeeze(1)
        
        # Second-order (FM)
        sparse_embed = self.embedding(x_sparse)
        second_order = self.fm(sparse_embed)
        
        # Deep
        sparse_flat = sparse_embed.view(-1, self.num_fields * self.embed_dim)
        if self.dense_dim > 0 and x_dense is not None:
            deep_input = torch.cat([sparse_flat, x_dense], dim=1)
        else:
            deep_input = sparse_flat
        deep_out = self.dnn(deep_input).squeeze(1)
        
        return self.global_bias + first_order + second_order + deep_out


# ============================================================
# 3. 학습 함수
# ============================================================

# ============================================================
# Loss Functions
# ============================================================

class RMSELoss(nn.Module):
    """기본 RMSE Loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


class WeightedMSELoss(nn.Module):
    """
    평균에서 먼 샘플에 더 큰 가중치를 부여하는 Weighted MSE Loss
    - 평균값으로 수렴하는 문제를 완화
    - alpha가 클수록 극단값 예측에 더 큰 페널티/보상
    """
    def __init__(self, mean_rating=7.2, alpha=1.5, use_rmse=True):
        super().__init__()
        self.mean_rating = mean_rating
        self.alpha = alpha
        self.use_rmse = use_rmse
    
    def forward(self, pred, target):
        # 평균에서 먼 타겟일수록 가중치 증가
        weight = 1.0 + self.alpha * torch.abs(target - self.mean_rating) / 10.0
        weighted_mse = (weight * (pred - target) ** 2).mean()
        
        if self.use_rmse:
            return torch.sqrt(weighted_mse)
        return weighted_mse


class FocalMSELoss(nn.Module):
    """
    Focal Loss 아이디어를 MSE에 적용
    - 예측이 틀릴수록(오차가 클수록) 더 큰 가중치
    - 쉬운 샘플(평균 근처)보다 어려운 샘플에 집중
    """
    def __init__(self, gamma=2.0, use_rmse=True):
        super().__init__()
        self.gamma = gamma
        self.use_rmse = use_rmse
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        # 오차가 클수록 가중치 증가 (정규화된 오차 사용)
        normalized_error = error / 10.0  # 1-10 스케일 가정
        weight = (1.0 + normalized_error) ** self.gamma
        focal_mse = (weight * (pred - target) ** 2).mean()
        
        if self.use_rmse:
            return torch.sqrt(focal_mse)
        return focal_mse


class QuantileWeightedLoss(nn.Module):
    """
    양 끝단(낮은/높은 평점) 예측을 강화하는 Loss
    - 분포의 꼬리 부분 예측 성능 향상
    """
    def __init__(self, low_q=0.3, high_q=0.7, use_rmse=True):
        super().__init__()
        self.low_q = low_q
        self.high_q = high_q
        self.use_rmse = use_rmse
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        
        # 낮은 평점 under-prediction 페널티
        low_penalty = torch.relu(target - pred) * self.low_q
        # 높은 평점 over-prediction 페널티  
        high_penalty = torch.relu(pred - target) * self.high_q
        
        combined = mse + low_penalty + high_penalty
        loss = combined.mean()
        
        if self.use_rmse:
            return torch.sqrt(loss)
        return loss


class HuberWeightedLoss(nn.Module):
    """
    Huber Loss + 평균 거리 가중치
    - 이상치에 robust하면서도 평균 수렴 방지
    """
    def __init__(self, delta=1.0, mean_rating=7.2, alpha=1.0, use_rmse=True):
        super().__init__()
        self.delta = delta
        self.mean_rating = mean_rating
        self.alpha = alpha
        self.use_rmse = use_rmse
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        
        # Huber Loss
        huber = torch.where(
            error <= self.delta,
            0.5 * error ** 2,
            self.delta * (error - 0.5 * self.delta)
        )
        
        # 평균 거리 가중치
        weight = 1.0 + self.alpha * torch.abs(target - self.mean_rating) / 10.0
        weighted_huber = (weight * huber).mean()
        
        if self.use_rmse:
            return torch.sqrt(weighted_huber)
        return weighted_huber


class CombinedLoss(nn.Module):
    """
    여러 Loss를 조합하여 사용
    - mse_weight: 기본 MSE 비중
    - weighted_weight: Weighted MSE 비중
    - quantile_weight: Quantile Loss 비중
    """
    def __init__(self, mean_rating=7.2, mse_weight=0.5, weighted_weight=0.3, 
                 quantile_weight=0.2, alpha=1.5, use_rmse=True):
        super().__init__()
        self.mean_rating = mean_rating
        self.mse_weight = mse_weight
        self.weighted_weight = weighted_weight
        self.quantile_weight = quantile_weight
        self.alpha = alpha
        self.use_rmse = use_rmse
    
    def forward(self, pred, target):
        # 기본 MSE
        mse = ((pred - target) ** 2).mean()
        
        # Weighted MSE
        weight = 1.0 + self.alpha * torch.abs(target - self.mean_rating) / 10.0
        weighted_mse = (weight * (pred - target) ** 2).mean()
        
        # Quantile penalty
        low_penalty = torch.relu(target - pred).mean() * 0.3
        high_penalty = torch.relu(pred - target).mean() * 0.3
        quantile = low_penalty + high_penalty
        
        combined = (self.mse_weight * mse + 
                   self.weighted_weight * weighted_mse + 
                   self.quantile_weight * quantile)
        
        if self.use_rmse:
            return torch.sqrt(combined)
        return combined


def get_loss_function(loss_type='weighted_mse', mean_rating=7.2, **kwargs):
    """
    손실 함수 선택 팩토리 함수
    
    Args:
        loss_type: 손실 함수 종류
            - 'rmse': 기본 RMSE
            - 'weighted_mse': 평균 거리 가중 MSE (기본값, 권장)
            - 'focal_mse': Focal MSE
            - 'quantile': Quantile 가중 Loss
            - 'huber': Huber + 가중치
            - 'combined': 여러 Loss 조합
        mean_rating: 학습 데이터의 평균 평점 (default: 7.2)
        **kwargs: 각 Loss 함수별 추가 파라미터
    
    Returns:
        nn.Module: 선택된 손실 함수
    
    Example:
        loss_fn = get_loss_function('weighted_mse', mean_rating=7.2, alpha=2.0)
        loss_fn = get_loss_function('focal_mse', gamma=3.0)
        loss_fn = get_loss_function('combined', mse_weight=0.4, weighted_weight=0.4)
    """
    loss_functions = {
        'rmse': lambda: RMSELoss(),
        'weighted_mse': lambda: WeightedMSELoss(
            mean_rating=mean_rating,
            alpha=kwargs.get('alpha', 1.5),
            use_rmse=kwargs.get('use_rmse', True)
        ),
        'focal_mse': lambda: FocalMSELoss(
            gamma=kwargs.get('gamma', 2.0),
            use_rmse=kwargs.get('use_rmse', True)
        ),
        'quantile': lambda: QuantileWeightedLoss(
            low_q=kwargs.get('low_q', 0.3),
            high_q=kwargs.get('high_q', 0.7),
            use_rmse=kwargs.get('use_rmse', True)
        ),
        'huber': lambda: HuberWeightedLoss(
            delta=kwargs.get('delta', 1.0),
            mean_rating=mean_rating,
            alpha=kwargs.get('alpha', 1.0),
            use_rmse=kwargs.get('use_rmse', True)
        ),
        'combined': lambda: CombinedLoss(
            mean_rating=mean_rating,
            mse_weight=kwargs.get('mse_weight', 0.5),
            weighted_weight=kwargs.get('weighted_weight', 0.3),
            quantile_weight=kwargs.get('quantile_weight', 0.2),
            alpha=kwargs.get('alpha', 1.5),
            use_rmse=kwargs.get('use_rmse', True)
        ),
    }
    
    if loss_type not in loss_functions:
        available = ', '.join(loss_functions.keys())
        raise ValueError(f"Unknown loss_type: {loss_type}. Available: {available}")
    
    return loss_functions[loss_type]()


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training', leave=False):
        x_sparse = batch['sparse'].to(device)
        x_dense = batch['dense'].to(device)
        y = batch['rating'].to(device)
        
        optimizer.zero_grad()
        pred = model(x_sparse, x_dense)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            x_sparse = batch['sparse'].to(device)
            x_dense = batch['dense'].to(device)
            y = batch['rating'].to(device)
            pred = model(x_sparse, x_dense)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting', leave=False):
            x_sparse = batch['sparse'].to(device)
            x_dense = batch['dense'].to(device)
            pred = model(x_sparse, x_dense)
            predictions.extend(pred.cpu().numpy())
    return np.array(predictions)


def apply_cold_start_postprocess(predictions, test_df, stats, cold_weight=0.5):
    """Cold Start 후처리"""
    predictions = predictions.copy()
    global_mean = stats['global_mean']
    author_stats_dict = stats['author_stats'].set_index('book_author')['author_rating_mean'].to_dict()
    
    is_cold_user = test_df['is_cold_user'].values
    is_cold_item = test_df['is_cold_item'].values
    authors = test_df['book_author'].values if 'book_author' in test_df.columns else [None] * len(test_df)
    
    adjusted_count = 0
    for i in range(len(predictions)):
        author = authors[i]
        author_mean = author_stats_dict.get(author, global_mean) if pd.notna(author) else global_mean
        
        if is_cold_user[i] and is_cold_item[i]:
            predictions[i] = author_mean
            adjusted_count += 1
        elif is_cold_item[i]:
            predictions[i] = (1 - cold_weight) * predictions[i] + cold_weight * author_mean
            adjusted_count += 1
    
    print(f"Cold Start 후처리: {adjusted_count}개 조정 ({adjusted_count/len(predictions)*100:.1f}%)")
    return np.clip(predictions, 1, 10)


# ============================================================
# 4. 메인 함수
# ============================================================

def main():
    # 설정
    DATA_PATH = 'data/'
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    EPOCHS = 8
    LEARNING_RATE = 2e-3
    EMBED_DIM = 32
    MLP_DIMS = [256, 128, 64]
    DROPOUT = 0.3
    VALID_RATIO = 0.2
    
    # 손실 함수 설정
    # 옵션: 'rmse', 'weighted_mse', 'focal_mse', 'quantile', 'huber', 'combined'
    LOSS_TYPE = 'weighted_mse'
    LOSS_KWARGS = {
        'alpha': 2.0,        # weighted_mse, huber: 평균 거리 가중치 강도
        # 'gamma': 2.0,      # focal_mse: focal 강도
        # 'low_q': 0.3,      # quantile: 낮은 평점 가중치
        # 'high_q': 0.7,     # quantile: 높은 평점 가중치
        # 'delta': 1.0,      # huber: Huber delta
        # 'mse_weight': 0.5, # combined: MSE 비중
        # 'weighted_weight': 0.3,  # combined: Weighted MSE 비중
        # 'quantile_weight': 0.2,  # combined: Quantile 비중
    }
    
    # 시드 고정
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print(f"Device: {DEVICE}")
    print(f"="*60)
    
    # ============ 데이터 로드 ============
    print("Loading data...")
    users = pd.read_csv(DATA_PATH + 'users.csv')
    books = pd.read_csv(DATA_PATH + 'books.csv')
    train = pd.read_csv(DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    
    print(f"Train: {len(train):,}, Test: {len(test):,}")
    
    # ============ 피처 엔지니어링 ============
    print("Feature engineering...")
    users_, books_, stats = process_enhanced_context_data(users, books, train, test)
    
    # 피처 정의
    sparse_cols = ['user_id', 'age_range', 'location_country', 'isbn', 'book_author', 'publisher', 'category', 'publication_range']
    dense_cols = ['user_rating_mean', 'user_rating_count_log', 'item_rating_mean', 'item_rating_count_log',
                  'author_rating_mean', 'author_rating_count_log', 'publisher_rating_mean', 'category_rating_mean']
    
    # 데이터 병합
    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    
    # Cold Start 플래그
    train_df['is_cold_user'] = 0
    train_df['is_cold_item'] = 0
    test_df['is_cold_user'] = (~test_df['user_id'].isin(stats['train_users'])).astype(int)
    test_df['is_cold_item'] = (~test_df['isbn'].isin(stats['train_items'])).astype(int)
    
    # Dense 피처 준비
    global_mean = stats['global_mean']
    for df in [train_df, test_df]:
        df['user_rating_count_log'] = np.log1p(df['user_rating_count'].fillna(0))
        df['item_rating_count_log'] = np.log1p(df['item_rating_count'].fillna(0))
        df['author_rating_count_log'] = np.log1p(df['author_rating_count'].fillna(0))
        df['user_rating_mean'] = df['user_rating_mean'].fillna(global_mean)
        df['item_rating_mean'] = df['item_rating_mean'].fillna(global_mean)
        df['author_rating_mean'] = df['author_rating_mean'].fillna(global_mean)
        df['publisher_rating_mean'] = df['publisher_rating_mean'].fillna(global_mean)
        df['category_rating_mean'] = df['category_rating_mean'].fillna(global_mean)
    
    # Sparse 피처 인코딩
    all_df = pd.concat([train_df, test_df], axis=0)
    label2idx = {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown').astype(str)
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].fillna('unknown').astype(str).map(label2idx[col])
        test_df[col] = test_df[col].fillna('unknown').astype(str).map(label2idx[col])
    
    field_dims = [len(label2idx[col]) for col in sparse_cols]
    print(f"Field dims: {field_dims}")
    print(f"Dense features: {len(dense_cols)}")
    
    # Dense 피처 정규화
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_df[dense_cols].fillna(0))
    test_dense = scaler.transform(test_df[dense_cols].fillna(0))
    
    # Train/Valid 분리
    X_train, X_valid, y_train, y_valid, train_dense_split, valid_dense_split = train_test_split(
        train_df[sparse_cols].values, train_df['rating'].values, train_dense,
        test_size=VALID_RATIO, random_state=SEED
    )
    
    print(f"Train: {len(X_train):,}, Valid: {len(X_valid):,}")
    
    # DataLoader
    train_dataset = EnhancedDataset(X_train, train_dense_split, y_train)
    valid_dataset = EnhancedDataset(X_valid, valid_dense_split, y_valid)
    test_dataset = EnhancedDataset(test_df[sparse_cols].values, test_dense, None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ============ 모델 ============
    print("="*60)
    print("Initializing model...")
    model = DeepFM_Enhanced(field_dims, len(dense_cols), EMBED_DIM, MLP_DIMS, DROPOUT).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ============ 학습 ============
    print("="*60)
    print("Training...")
    
    loss_fn = get_loss_function(LOSS_TYPE, mean_rating=global_mean, **LOSS_KWARGS)
    print(f"Loss function: {LOSS_TYPE} (mean_rating={global_mean:.2f})")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss = validate(model, valid_loader, loss_fn, DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} - Train RMSE: {train_loss:.4f}, Valid RMSE: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  → Best model saved! (Valid RMSE: {val_loss:.4f})")
    
    # Best 모델 로드
    model.load_state_dict(best_model_state)
    print(f"\nBest Valid RMSE: {best_val_loss:.4f}")
    
    # ============ 예측 ============
    print("="*60)
    print("Predicting...")
    
    predictions = predict(model, test_loader, DEVICE)
    
    # Cold Start 후처리
    predictions = apply_cold_start_postprocess(predictions, test_df, stats, cold_weight=0.5)
    
    print(f"Predictions - min: {predictions.min():.2f}, max: {predictions.max():.2f}, mean: {predictions.mean():.2f}")
    
    # ============ 저장 ============
    os.makedirs('saved/submit', exist_ok=True)
    submission = sub.copy()
    submission['rating'] = predictions
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved/submit/{timestamp}_DeepFM_Enhanced.csv"
    submission.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")
    
    print("="*60)
    print("Done!")
    
    return predictions


if __name__ == "__main__":
    main()