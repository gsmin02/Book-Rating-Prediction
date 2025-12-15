"""
grid_search.py - DeepFM_Enhanced 하이퍼파라미터 그리드서치

사용법:
    python grid_search.py

탐색할 하이퍼파라미터 범위를 SEARCH_SPACE에서 수정하세요.
"""

import os
import itertools
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
import json


# ============================================================
# 🔧 그리드서치 설정 - 여기만 수정하면 됩니다!
# ============================================================

FIX = True
UNFIX = False

SEARCH_SPACE = {
    'LEARNING_RATE': [3e-3] if UNFIX else [1e-3, 3e-3, 6e-3, 1e-2],
    'EMBED_DIM': [32] if UNFIX else [16, 32, 64],
    'MLP_DIMS': [[256, 128, 64]] if FIX else [
        [64, 32],
        [128, 64, 32],
        [256, 128, 64],
        [256, 128, 64, 32],
    ],

    'DROPOUT': [0.1] if FIX else [0.1, 0.2, 0.3, 0.4],
    'BATCH_SIZE': [64] if UNFIX else [16, 32, 64, 128],
}

# 고정 파라미터
FIXED_PARAMS = {
    'EPOCHS': 8,          # 그리드서치 시 에폭 수 (빠른 탐색을 위해 줄일 수 있음)
    'VALID_RATIO': 0.2,
    'SEED': 42,
    'DATA_PATH': 'data/',
}

# 결과 저장 경로
RESULTS_DIR = 'grid_search_results'


# ============================================================
# 데이터 처리 함수들 (run_enhanced.py에서 가져옴)
# ============================================================

def str2list(x: str) -> list:
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
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
    users_ = users.copy()
    books_ = books.copy()
    
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else 'unknown')
    books_['language'] = books_['language'].fillna('en')
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)
    books_.loc[books_['year_of_publication'] < 1900, 'year_of_publication'] = 1996
    
    users_['age'] = users_['age'].fillna(34)
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)
    users_['location_list'] = users_['location'].apply(lambda x: split_location(x))
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    
    global_mean = train_df['rating'].mean()
    global_std = train_df['rating'].std()
    
    user_stats = train_df.groupby('user_id').agg(
        user_rating_mean=('rating', 'mean'),
        user_rating_count=('rating', 'count')
    ).reset_index()
    
    item_stats = train_df.groupby('isbn').agg(
        item_rating_mean=('rating', 'mean'),
        item_rating_count=('rating', 'count')
    ).reset_index()
    
    train_with_author = train_df.merge(books_[['isbn', 'book_author']], on='isbn', how='left')
    author_stats = train_with_author.groupby('book_author').agg(
        author_rating_mean=('rating', 'mean'),
        author_rating_count=('rating', 'count')
    ).reset_index()
    
    train_with_publisher = train_df.merge(books_[['isbn', 'publisher']], on='isbn', how='left')
    publisher_stats = train_with_publisher.groupby('publisher').agg(
        publisher_rating_mean=('rating', 'mean'),
    ).reset_index()
    
    train_with_category = train_df.merge(books_[['isbn', 'category']], on='isbn', how='left')
    category_stats = train_with_category.groupby('category').agg(
        category_rating_mean=('rating', 'mean'),
    ).reset_index()
    
    train_users = set(train_df['user_id'].unique())
    train_items = set(train_df['isbn'].unique())
    
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
# 모델 정의
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
        
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FMLayer()
        
        if dense_dim > 0:
            self.dense_linear = nn.Linear(dense_dim, 1)
        
        deep_input_dim = (embed_dim * self.num_fields) + dense_dim
        self.dnn = MLP(deep_input_dim, mlp_dims, dropout)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse, x_dense=None):
        first_order = self.linear(x_sparse).squeeze(1)
        if self.dense_dim > 0 and x_dense is not None:
            first_order = first_order + self.dense_linear(x_dense).squeeze(1)
        
        sparse_embed = self.embedding(x_sparse)
        second_order = self.fm(sparse_embed)
        
        sparse_flat = sparse_embed.view(-1, self.num_fields * self.embed_dim)
        if self.dense_dim > 0 and x_dense is not None:
            deep_input = torch.cat([sparse_flat, x_dense], dim=1)
        else:
            deep_input = sparse_flat
        deep_out = self.dnn(deep_input).squeeze(1)
        
        return self.global_bias + first_order + second_order + deep_out


# ============================================================
# 학습 함수
# ============================================================

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
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
        for batch in dataloader:
            x_sparse = batch['sparse'].to(device)
            x_dense = batch['dense'].to(device)
            y = batch['rating'].to(device)
            pred = model(x_sparse, x_dense)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


# ============================================================
# 그리드서치 핵심 함수
# ============================================================

def run_single_experiment(params, train_loader, valid_loader, field_dims, dense_dim, device, epochs):
    """단일 하이퍼파라미터 조합으로 실험 실행"""
    
    model = DeepFM_Enhanced(
        field_dims=field_dims,
        dense_dim=dense_dim,
        embed_dim=params['EMBED_DIM'],
        mlp_dims=params['MLP_DIMS'],
        dropout=params['DROPOUT']
    ).to(device)
    
    loss_fn = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, valid_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return {
        'best_val_rmse': best_val_loss,
        'final_train_rmse': train_losses[-1],
        'final_val_rmse': val_losses[-1],
        'train_history': train_losses,
        'val_history': val_losses,
    }


def prepare_data(data_path, valid_ratio, seed):
    """데이터 준비 (한 번만 실행)"""
    print("Loading and preparing data...")
    
    users = pd.read_csv(data_path + 'users.csv')
    books = pd.read_csv(data_path + 'books.csv')
    train = pd.read_csv(data_path + 'train_ratings.csv')
    test = pd.read_csv(data_path + 'test_ratings.csv')
    
    users_, books_, stats = process_enhanced_context_data(users, books, train, test)
    
    sparse_cols = ['user_id', 'age_range', 'location_country', 'isbn', 'book_author', 
                   'publisher', 'category', 'publication_range']
    dense_cols = ['user_rating_mean', 'user_rating_count_log', 'item_rating_mean', 
                  'item_rating_count_log', 'author_rating_mean', 'author_rating_count_log', 
                  'publisher_rating_mean', 'category_rating_mean']
    
    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    
    global_mean = stats['global_mean']
    train_df['user_rating_count_log'] = np.log1p(train_df['user_rating_count'].fillna(0))
    train_df['item_rating_count_log'] = np.log1p(train_df['item_rating_count'].fillna(0))
    train_df['author_rating_count_log'] = np.log1p(train_df['author_rating_count'].fillna(0))
    train_df['user_rating_mean'] = train_df['user_rating_mean'].fillna(global_mean)
    train_df['item_rating_mean'] = train_df['item_rating_mean'].fillna(global_mean)
    train_df['author_rating_mean'] = train_df['author_rating_mean'].fillna(global_mean)
    train_df['publisher_rating_mean'] = train_df['publisher_rating_mean'].fillna(global_mean)
    train_df['category_rating_mean'] = train_df['category_rating_mean'].fillna(global_mean)
    
    # Sparse 피처 인코딩
    label2idx = {}
    for col in sparse_cols:
        train_df[col] = train_df[col].fillna('unknown').astype(str)
        unique_labels = train_df[col].astype("category").cat.categories
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].map(label2idx[col])
    
    field_dims = [len(label2idx[col]) for col in sparse_cols]
    
    # Dense 피처 정규화
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_df[dense_cols].fillna(0))
    
    # Train/Valid 분리
    X_train, X_valid, y_train, y_valid, train_dense_split, valid_dense_split = train_test_split(
        train_df[sparse_cols].values, train_df['rating'].values, train_dense,
        test_size=valid_ratio, random_state=seed
    )
    
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_valid': y_valid,
        'train_dense': train_dense_split,
        'valid_dense': valid_dense_split,
        'field_dims': field_dims,
        'dense_dim': len(dense_cols),
    }


def run_grid_search():
    """그리드서치 메인 함수"""
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("="*70)
    
    # 시드 고정
    torch.manual_seed(FIXED_PARAMS['SEED'])
    np.random.seed(FIXED_PARAMS['SEED'])
    
    # 데이터 준비 (한 번만)
    data = prepare_data(
        FIXED_PARAMS['DATA_PATH'], 
        FIXED_PARAMS['VALID_RATIO'], 
        FIXED_PARAMS['SEED']
    )
    
    # 모든 하이퍼파라미터 조합 생성
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    all_combinations = list(itertools.product(*param_values))
    
    total_experiments = len(all_combinations)
    print(f"\n총 {total_experiments}개의 하이퍼파라미터 조합을 테스트합니다.")
    print(f"탐색 공간: {SEARCH_SPACE}")
    print("="*70)
    
    # 결과 저장
    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 그리드서치 실행
    for idx, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))
        
        print(f"\n[{idx}/{total_experiments}] 실험 중...")
        print(f"  Parameters: {params}")
        
        # 배치 사이즈에 따라 DataLoader 재생성
        train_dataset = EnhancedDataset(data['X_train'], data['train_dense'], data['y_train'])
        valid_dataset = EnhancedDataset(data['X_valid'], data['valid_dense'], data['y_valid'])
        
        train_loader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
        
        # 실험 실행
        try:
            result = run_single_experiment(
                params=params,
                train_loader=train_loader,
                valid_loader=valid_loader,
                field_dims=data['field_dims'],
                dense_dim=data['dense_dim'],
                device=device,
                epochs=FIXED_PARAMS['EPOCHS']
            )
            
            # 결과 기록
            result_entry = {
                **params,
                'MLP_DIMS': str(params['MLP_DIMS']),  # JSON 저장을 위해 문자열로 변환
                'best_val_rmse': result['best_val_rmse'],
                'final_train_rmse': result['final_train_rmse'],
                'final_val_rmse': result['final_val_rmse'],
            }
            results.append(result_entry)
            
            print(f"  ✓ Best Valid RMSE: {result['best_val_rmse']:.4f}")
            
        except Exception as e:
            print(f"  ✗ 실험 실패: {e}")
            continue
    
    # 결과 정리 및 저장
    print("\n" + "="*70)
    print("그리드서치 완료!")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_rmse')
    
    # 결과 출력
    print("\n📊 상위 10개 결과:")
    print(results_df.head(10).to_string(index=False))
    
    # 최적 파라미터
    best_params = results_df.iloc[0].to_dict()
    print("\n🏆 최적 하이퍼파라미터:")
    for key, value in best_params.items():
        if key not in ['best_val_rmse', 'final_train_rmse', 'final_val_rmse']:
            print(f"  {key}: {value}")
    print(f"  → Best Valid RMSE: {best_params['best_val_rmse']:.4f}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV 저장
    csv_path = f"{RESULTS_DIR}/grid_search_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📁 결과 저장: {csv_path}")
    
    # 최적 파라미터 JSON 저장
    json_path = f"{RESULTS_DIR}/best_params_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"📁 최적 파라미터 저장: {json_path}")
    
    return results_df, best_params


if __name__ == "__main__":
    results_df, best_params = run_grid_search()
