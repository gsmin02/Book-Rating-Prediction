"""
Enhanced Context Data Loader
- 전략 2: 수치형 피처 직접 사용 (user_bias, item_bias, author_mean 등)
- 전략 3: Cold Start 플래그 및 대체값 준비

EDA 핵심 발견 반영:
1. Author 평균 rating → 14% RMSE 개선 효과
2. User/Item bias → 개인 성향 차이 큼 (4.5점 차이)
3. Cold Item 25.8% → Author로 62% 커버 가능
"""

import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
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
    """
    EDA 발견사항을 반영한 피처 엔지니어링
    
    추가되는 피처:
    - user_rating_mean: User 평균 rating (bias)
    - user_rating_count: User activity
    - item_rating_mean: Item 평균 rating (bias)
    - item_rating_count: Item popularity
    - author_rating_mean: Author 평균 rating (핵심!)
    - author_rating_count: Author의 총 rating 수
    - is_cold_user: Cold User 플래그
    - is_cold_item: Cold Item 플래그
    """
    users_ = users.copy()
    books_ = books.copy()
    
    # ============ 기존 전처리 (context_data.py 동일) ============
    # Books 전처리
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else 'unknown')
    books_['language'] = books_['language'].fillna('en')  # 최빈값
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)
    
    # 이상치 처리 (EDA 발견: 1376, 1378년 등)
    books_.loc[books_['year_of_publication'] < 1900, 'year_of_publication'] = 1996  # 중앙값
    
    # Users 전처리
    users_['age'] = users_['age'].fillna(34)  # 중앙값 (EDA 결과)
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)
    
    # Location 파싱
    users_['location_list'] = users_['location'].apply(lambda x: split_location(x))
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    
    # ============ 전략 2: 수치형 피처 생성 (Train에서만 계산!) ============
    global_mean = train_df['rating'].mean()
    global_std = train_df['rating'].std()
    
    # 1. User 통계 (Train에서만)
    user_stats = train_df.groupby('user_id').agg(
        user_rating_mean=('rating', 'mean'),
        user_rating_std=('rating', 'std'),
        user_rating_count=('rating', 'count')
    ).reset_index()
    user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(global_std)
    
    # 2. Item 통계 (Train에서만)
    item_stats = train_df.groupby('isbn').agg(
        item_rating_mean=('rating', 'mean'),
        item_rating_std=('rating', 'std'),
        item_rating_count=('rating', 'count')
    ).reset_index()
    item_stats['item_rating_std'] = item_stats['item_rating_std'].fillna(global_std)
    
    # 3. Author 통계 (Train에서만) - EDA 핵심 발견!
    train_with_author = train_df.merge(books_[['isbn', 'book_author']], on='isbn', how='left')
    author_stats = train_with_author.groupby('book_author').agg(
        author_rating_mean=('rating', 'mean'),
        author_rating_std=('rating', 'std'),
        author_rating_count=('rating', 'count')
    ).reset_index()
    author_stats['author_rating_std'] = author_stats['author_rating_std'].fillna(global_std)
    
    # 4. Publisher 통계 (Train에서만)
    train_with_publisher = train_df.merge(books_[['isbn', 'publisher']], on='isbn', how='left')
    publisher_stats = train_with_publisher.groupby('publisher').agg(
        publisher_rating_mean=('rating', 'mean'),
        publisher_rating_count=('rating', 'count')
    ).reset_index()
    
    # 5. Category 통계 (Train에서만)
    train_with_category = train_df.merge(books_[['isbn', 'category']], on='isbn', how='left')
    category_stats = train_with_category.groupby('category').agg(
        category_rating_mean=('rating', 'mean'),
        category_rating_count=('rating', 'count')
    ).reset_index()
    
    # ============ 전략 3: Cold Start 식별 ============
    train_users = set(train_df['user_id'].unique())
    train_items = set(train_df['isbn'].unique())
    train_authors = set(author_stats['book_author'].dropna().unique())
    
    # 통계 정보를 users_, books_에 병합
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
        'train_authors': train_authors,
        'user_stats': user_stats,
        'item_stats': item_stats,
        'author_stats': author_stats,
        'publisher_stats': publisher_stats,
        'category_stats': category_stats,
    }


class EnhancedDataset(Dataset):
    """
    Sparse(범주형) + Dense(수치형) 피처를 함께 처리하는 Dataset
    """
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


def enhanced_context_data_load(args):
    """
    Enhanced Context Data 로드 함수
    - 범주형(sparse) + 수치형(dense) 피처 분리
    - Cold Start 정보 포함
    """
    # 데이터 로드
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')
    
    # Enhanced 전처리
    users_, books_, stats = process_enhanced_context_data(users, books, train, test)
    
    # ============ 피처 정의 ============
    # Sparse 피처 (범주형 → Embedding)
    user_sparse_features = ['user_id', 'age_range', 'location_country']
    book_sparse_features = ['isbn', 'book_author', 'publisher', 'category', 'publication_range']
    sparse_cols = user_sparse_features + book_sparse_features
    
    # Dense 피처 (수치형 → 직접 사용)
    dense_cols = [
        'user_rating_mean', 'user_rating_count_log',
        'item_rating_mean', 'item_rating_count_log',
        'author_rating_mean', 'author_rating_count_log',
        'publisher_rating_mean', 'category_rating_mean',
    ]
    
    # ============ 데이터 병합 ============
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')
    
    # ============ Cold Start 플래그 ============
    train_df['is_cold_user'] = 0
    train_df['is_cold_item'] = 0
    test_df['is_cold_user'] = (~test_df['user_id'].isin(stats['train_users'])).astype(int)
    test_df['is_cold_item'] = (~test_df['isbn'].isin(stats['train_items'])).astype(int)
    
    # ============ Dense 피처 전처리 ============
    global_mean = stats['global_mean']
    global_std = stats['global_std']
    
    # Log 변환 (count 피처)
    for df in [train_df, test_df]:
        df['user_rating_count_log'] = np.log1p(df['user_rating_count'].fillna(0))
        df['item_rating_count_log'] = np.log1p(df['item_rating_count'].fillna(0))
        df['author_rating_count_log'] = np.log1p(df['author_rating_count'].fillna(0))
    
    # Cold Start 대체값 (전략 3)
    # Cold User → 전체 평균
    for df in [train_df, test_df]:
        df['user_rating_mean'] = df['user_rating_mean'].fillna(global_mean)
        df['item_rating_mean'] = df['item_rating_mean'].fillna(global_mean)
        # Cold Item인데 Author는 있는 경우 → Author 평균 사용
        mask_cold_item_with_author = df['item_rating_mean'].isna() & df['author_rating_mean'].notna()
        df.loc[mask_cold_item_with_author, 'item_rating_mean'] = df.loc[mask_cold_item_with_author, 'author_rating_mean']
        # 그래도 없으면 전체 평균
        df['author_rating_mean'] = df['author_rating_mean'].fillna(global_mean)
        df['publisher_rating_mean'] = df['publisher_rating_mean'].fillna(global_mean)
        df['category_rating_mean'] = df['category_rating_mean'].fillna(global_mean)
    
    # ============ Sparse 피처 라벨 인코딩 ============
    all_df = pd.concat([train_df, test_df], axis=0)
    
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown').astype(str)
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].fillna('unknown').astype(str).map(label2idx[col])
        test_df[col] = test_df[col].fillna('unknown').astype(str).map(label2idx[col])
    
    field_dims = [len(label2idx[col]) for col in sparse_cols]
    
    # ============ Dense 피처 정규화 ============
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_df[dense_cols].fillna(0))
    test_dense = scaler.transform(test_df[dense_cols].fillna(0))
    
    # DataFrame에 정규화된 값 저장
    for i, col in enumerate(dense_cols):
        train_df[f'{col}_scaled'] = train_dense[:, i]
        test_df[f'{col}_scaled'] = test_dense[:, i]
    
    dense_cols_scaled = [f'{col}_scaled' for col in dense_cols]
    
    data = {
        'train': train_df,
        'test': test_df,
        'field_names': sparse_cols,
        'field_dims': field_dims,
        'dense_cols': dense_cols_scaled,
        'dense_dim': len(dense_cols),
        'label2idx': label2idx,
        'idx2label': idx2label,
        'sub': sub,
        'stats': stats,
        'scaler': scaler,
    }
    
    return data


def enhanced_context_data_split(args, data):
    """Train/Valid 분리"""
    from sklearn.model_selection import train_test_split
    
    if args.dataset.valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            data['train'].drop(['rating'], axis=1),
            data['train']['rating'],
            test_size=args.dataset.valid_ratio,
            random_state=args.seed,
            shuffle=True
        )
        data['X_train'], data['X_valid'] = X_train, X_valid
        data['y_train'], data['y_valid'] = y_train, y_valid
    
    return data


def enhanced_context_data_loader(args, data):
    """DataLoader 생성"""
    sparse_cols = data['field_names']
    dense_cols = data['dense_cols']
    
    # Train Dataset
    train_dataset = EnhancedDataset(
        sparse_data=data['X_train'][sparse_cols].values,
        dense_data=data['X_train'][dense_cols].values,
        rating=data['y_train'].values
    )
    
    # Valid Dataset
    if args.dataset.valid_ratio != 0:
        valid_dataset = EnhancedDataset(
            sparse_data=data['X_valid'][sparse_cols].values,
            dense_data=data['X_valid'][dense_cols].values,
            rating=data['y_valid'].values
        )
    else:
        valid_dataset = None
    
    # Test Dataset
    test_dataset = EnhancedDataset(
        sparse_data=data['test'][sparse_cols].values,
        dense_data=data['test'][dense_cols].values,
        rating=None
    )
    
    # DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.dataloader.batch_size, 
        shuffle=args.dataloader.shuffle, 
        num_workers=args.dataloader.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.dataloader.batch_size, 
        shuffle=False, 
        num_workers=args.dataloader.num_workers
    ) if valid_dataset else None
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.dataloader.batch_size, 
        shuffle=False, 
        num_workers=args.dataloader.num_workers
    )
    
    data['train_dataloader'] = train_dataloader
    data['valid_dataloader'] = valid_dataloader
    data['test_dataloader'] = test_dataloader
    
    return data
