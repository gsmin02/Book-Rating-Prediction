import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
import os

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')


def normalize_location_element(element):
        """리스트 요소를 정제 (공백 제거, 소문자 변환, 특수문자 제거)"""
        if not isinstance(element, str):
            return ''
        
        # 1. 공백 제거 및 소문자 변환
        res = element.strip().lower()
        
        # 2. 특수 문자 제거 (a-z, A-Z, /, 공백만 남김)
        # regex.sub(r'[^a-z/ ]', '', res)를 사용합니다. 대문자는 이미 소문자로 변환했으므로 a-z만 확인
        res = regex.sub(r'[^a-z/ ]', '', res) 

        return res.strip() # 최종적으로 다시 한번 앞뒤 공백 제거 (특수문자 제거로 인해 생길 수 있음)

def normalize_category_element(element):
        """리스트 요소를 정제 (공백 제거, 소문자 변환, 특수문자 제거)"""
        if not isinstance(element, str):
            return ''
        
        # 소문자 + 공백 제거
        res = element.strip().lower()
        
        # 특수 문자 제거 (a-z, A-Z, /, 공백만 남김)
        res = regex.sub(r'[^a-z/ ]', '', res)

        # 'books' 제거
        res = res.replace('books', ' ').strip()
    
        # 'book' 제거
        res = res.replace('book', ' ').strip()
        
        # 여러 개의 공백을 하나의 공백으로 치환 (예: "fantasy  fiction" -> "fantasy fiction")
        res = regex.sub(r'\s+', ' ', res).strip()

        return res

def handle_rare(df: pd.DataFrame, 
                                        feature_col: str, 
                                        min_count: int, 
                                        replacement_name: str) -> pd.DataFrame:
    """
    주어진 컬럼에서 사용자(user_id) 수가 min_count 미만인 범주를 replacement_name으로 통합합니다.
    (이 함수는 users_df_cleaned와 같은 사용자 레벨 데이터프레임에 적용됩니다.)
    """
    
    if feature_col not in df.columns:
        print(f"오류: 데이터프레임에 '{feature_col}' 컬럼이 존재하지 않습니다.")
        return df

    category_counts = df[feature_col].value_counts()
    
    rare_categories = category_counts[category_counts < min_count].index.tolist()
    
    df.loc[df[feature_col].isin(rare_categories), feature_col] = replacement_name
    
    print(f"\n--- 희소성 처리 결과: {feature_col} ---")
    print(f"총 {len(rare_categories)}개의 희소 범주가 '{replacement_name}'으로 통합되었습니다.")
    print(f"통합 후 고유 범주 개수: {df[feature_col].nunique()}개")
    
    return df
    
def process_context_data(users, books, train):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    
    Returns
    -------
    label_to_idx : dict
        데이터를 인덱싱한 정보를 담은 딕셔너리
    idx_to_label : dict
        인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
    train_df : pd.DataFrame
        train 데이터
    test_df : pd.DataFrame
        test 데이터
    """

    users_ = users.copy()
    books_ = books.copy()

    # 데이터 전처리

    # publication 정보 전처리 ===========================================================================
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)  # 1990년대, 2000년대, 2010년대, ...

    CURRENT_YEAR = 2025
    books_['book_age'] = CURRENT_YEAR - books_['year_of_publication']
    books_.loc[(books_['book_age'] < 0) | (books_['book_age'] > 300), 'book_age'] = np.nan
    books_['book_age'] = books_['book_age'].fillna(books_['book_age'].mean())
    # =================================================================================================

    # language 정보 전처리 ==============================================================================
    MIN_BOOKS_LANGUAGE = 15      # 언어 15개 미만
    books_['language'] = books_['language'].apply(lambda x: normalize_category_element(x))
    books_ = handle_rare(books_, 'language', min_count=MIN_BOOKS_LANGUAGE, replacement_name='RARE_LANGUAGE')
    
    books_['language'] = books_['language'].fillna('UNKNOWN_LANGUAGE')
    # =================================================================================================

    # book_title 정보 전처리 ==============================================================================
    books_['book_title_length'] = books_['book_title'].astype(str).apply(lambda x: len(x))
    books_.loc[books_['book_title_length'] == 0, 'book_title_length'] = np.nan
    books_['book_title_length'] = books_['book_title_length'].fillna(books_['book_title_length'].mean())

    books_['book_title_word_count'] = books_['book_title'].astype(str).apply(lambda x: len(x.split()))
    books_.loc[books_['book_title_word_count'] == 0, 'book_title_word_count'] = np.nan
    books_['book_title_word_count'] = books_['book_title_word_count'].fillna(books_['book_title_word_count'].mean())
    # =================================================================================================

    # summary 정보 전처리 ==============================================================================
    books_['summary'] = books_['summary'].replace('nan', np.nan)
    books_['has_summary'] = books_['summary'].notna().astype(int)
    # =================================================================================================

    # text 정보 전처리 =================================================================================
    print('\nLoading Text Embeddings for Fusion...')
    PCA_DIM = 8
    try:
        book_summary_vector_list = np.load('./data/text_vector/book_summary_vector.npy', allow_pickle=True)
        user_summary_merge_vector_list = np.load('./data/text_vector/user_summary_merge_vector.npy', allow_pickle=True)
    except FileNotFoundError:
        print("FATAL ERROR: 텍스트 벡터 파일(.npy)을 찾을 수 없습니다.")
        return

    book_summary_vector_df = pd.DataFrame({'isbn': book_summary_vector_list[:, 0]})
    book_summary_vector_df['book_summary_vector'] = list(book_summary_vector_list[:, 1:].astype(np.float32))

    user_summary_vector_df = pd.DataFrame({'user_id': user_summary_merge_vector_list[:, 0]})
    user_summary_vector_df['user_summary_merge_vector'] = list(user_summary_merge_vector_list[:, 1:].astype(np.float32))

    print(f"\nApplying PCA to Book Summary Vectors (768D -> {PCA_DIM}D)...")
    book_vectors = np.array(book_summary_vector_df['book_summary_vector'].tolist())
    pca_book = PCA(n_components=PCA_DIM)
    book_vectors_pca = pca_book.fit_transform(book_vectors)

    # 축소된 벡터를 DataFrame에 업데이트
    book_summary_vector_df['book_summary_vector'] = list(book_vectors_pca)

    print(f"Applying PCA to User Summary Vectors (768D -> {PCA_DIM}D)...")
    user_vectors = np.array(user_summary_vector_df['user_summary_merge_vector'].tolist())
    pca_user = PCA(n_components=PCA_DIM)
    user_vectors_pca = pca_user.fit_transform(user_vectors)

    # 축소된 벡터를 DataFrame에 업데이트
    user_summary_vector_df['user_summary_merge_vector'] = list(user_vectors_pca)

    books_ = pd.merge(books_, book_summary_vector_df, on='isbn', how='left')
    users_ = pd.merge(users_, user_summary_vector_df, on='user_id', how='left')
    # =================================================================================================

    # location 정보 전처리 ==============================================================================
    # location ,로 나누기
    users_['location_list'] = users_['location'].str.split(',')

    # 길이가 3인 것(모든 위치 기입한 정보만) + Null과 Black 제외 시킨 데이터만 남기기
    condition_length = users_['location_list'].apply(lambda x: len(x) == 3)
    invalid_values = {'n/a', ''}
    condition_no_na_or_empty = users_['location_list'].apply(
        lambda x: not any(item.strip().lower() in invalid_values for item in x)
    )
    users_df_cleaned = users_[condition_length & condition_no_na_or_empty].copy()

    # 각 리스트의 요소를 정규화
    users_df_cleaned['location_list_normalized'] = users_df_cleaned['location_list'].apply(
        lambda x: [normalize_location_element(i) for i in x]
        )
    
    # 4-2. 정규화된 리스트를 사용하여 최종 피처 추출
    users_df_cleaned['location_city'] = users_df_cleaned['location_list_normalized'].apply(lambda x: x[0])
    users_df_cleaned['location_state'] = users_df_cleaned['location_list_normalized'].apply(lambda x: x[1])
    users_df_cleaned['location_country'] = users_df_cleaned['location_list_normalized'].apply(lambda x: x[2])
    
    MIN_USERS_COUNTRY = 50  # 국가: 사용자 50명 미만
    MIN_USERS_STATE = 30     # 주: 사용자 30명 미만
    MIN_USERS_CITY = 100      # 도시: 사용자 20명 미만

    users_df_cleaned = handle_rare(users_df_cleaned, 'location_country', min_count=MIN_USERS_COUNTRY, replacement_name='RARE_COUNTRY')
    users_df_cleaned = handle_rare(users_df_cleaned, 'location_state', min_count=MIN_USERS_STATE, replacement_name='RARE_STATE')
    users_df_cleaned = handle_rare(users_df_cleaned, 'location_city', min_count=MIN_USERS_CITY, replacement_name='RARE_CITY')

    location_features = ['user_id', 'location_country', 'location_state', 'location_city']
    users_ = users_.drop(['location', 'location_list'], axis=1, errors='ignore')
    users_ = users_.merge(
        users_df_cleaned[location_features],
        on='user_id',
        how='left'
    )
    users_['location_country'] = users_['location_country'].fillna('UNKNOWN_COUNTRY')
    users_['location_state'] = users_['location_state'].fillna('UNKNOWN_STATE')
    users_['location_city'] = users_['location_city'].fillna('UNKNOWN_CITY')
    # =================================================================================================

    # user + book + rating 정보 전처리 =========================================================================
    ratings_explicit = train[train['rating'] != 0].copy()

    MIN_RATINGS = 3

    global_avg_rating = ratings_explicit['rating'].mean()
    global_std_rating = ratings_explicit['rating'].std()

    # user 관련 새로운 정보를 포함한 df 만들기
    user_stats_df = ratings_explicit.groupby('user_id')['rating'].agg(
        user_avg_rating='mean',        # 사용자 평균 평점 (사용자 편향)
        user_rating_count='count',     # 사용자 평점 횟수 (활동성)
        user_rating_std='std',          # 사용자 평점 편차 (감정적)
        user_rating_skewness=skew,     # 왜도 (얼마나 치우쳤는지)
        user_rating_kurtosis=kurtosis, # 첨도 (얼마나 뾰족한지)
        user_median_rating='median', # 중앙값
        user_mode_rating=lambda x: x.mode()[0] if not x.mode().empty else np.nan # 최빈값
    ).reset_index()

    user_stats_df['user_rating_cv'] = (user_stats_df['user_rating_std'] / user_stats_df['user_avg_rating']).fillna(0)

    mask = user_stats_df['user_rating_count'] < MIN_RATINGS
    user_stats_df.loc[mask, 'user_rating_skewness'] = np.nan
    user_stats_df.loc[mask, 'user_rating_kurtosis'] = np.nan
    user_stats_df.loc[mask, 'user_mode_rating'] = np.nan

    user_stats_df['user_global_bias'] = user_stats_df['user_avg_rating'] - global_avg_rating

    users_ = users_.merge(user_stats_df,on='user_id',how='left')

    # book 관련 새로운 정보를 포함한 df 만들기
    book_stats_df = ratings_explicit.groupby('isbn')['rating'].agg(
        book_avg_rating='mean',       # 책 평균 평점 (책 편향)
        book_rating_count='count',     # 책 평점 받은 횟수 (책 인기도)
        book_rating_std='std',          # 책 평점 편차 (책 호불호)
        book_rating_skewness=skew,     # 왜도 (얼마나 치우쳤는지)
        book_rating_kurtosis=kurtosis, # 첨도 (얼마나 뾰족한지)
        book_median_rating='median',    # 중앙값
        book_mode_rating=lambda x: x.mode()[0] if not x.mode().empty else np.nan # 최빈값
    ).reset_index()

    book_stats_df['book_rating_cv'] = (book_stats_df['book_rating_std'] / book_stats_df['book_avg_rating']).fillna(0)

    mask = book_stats_df['book_rating_count'] < MIN_RATINGS
    book_stats_df.loc[mask, 'book_rating_skewness'] = np.nan
    book_stats_df.loc[mask, 'book_rating_kurtosis'] = np.nan
    book_stats_df.loc[mask, 'book_mode_rating'] = np.nan

    book_stats_df['book_global_bias'] = book_stats_df['book_avg_rating'] - global_avg_rating

    books_ = books_.merge(book_stats_df,on='isbn',how='left')

    def categorize_user_activity(count):
        if count <= 1: return 'RARE_USER'
        elif count <= 5: return 'MEDIUM_USER'
        else: return 'ACTIVE_USER'

    users_['user_activity_level'] = users_['user_rating_count'].apply(categorize_user_activity)
    users_['user_avg_rating'] = users_['user_avg_rating'].fillna(global_avg_rating)
    users_['user_rating_std'] = users_['user_rating_std'].fillna(global_std_rating)
    users_['user_rating_count'] = users_['user_rating_count'].fillna(0)
    users_['user_rating_skewness'] = users_['user_rating_skewness'].fillna(0)
    users_['user_rating_kurtosis'] = users_['user_rating_kurtosis'].fillna(0)
    users_['user_median_rating'] = users_['user_median_rating'].fillna(global_avg_rating)
    users_['user_mode_rating'] = users_['user_mode_rating'].fillna(global_avg_rating) 
    users_['user_rating_cv'] = users_['user_rating_cv'].fillna(0)
    
    def categorize_book_rarity(count):
        if count <= 1: return 'RARE_BOOK'
        else: return 'POPULAR_BOOK'

    books_['book_rarity_level'] = books_['book_rating_count'].apply(categorize_book_rarity)
    books_['book_avg_rating'] = books_['book_avg_rating'].fillna(global_avg_rating)
    books_['book_rating_std'] = books_['book_rating_std'].fillna(global_std_rating)
    books_['book_rating_count'] = books_['book_rating_count'].fillna(0)
    books_['book_rating_skewness'] = books_['book_rating_skewness'].fillna(0)
    books_['book_rating_kurtosis'] = books_['book_rating_kurtosis'].fillna(0)
    books_['book_median_rating'] = books_['book_median_rating'].fillna(global_avg_rating)
    books_['book_mode_rating'] = books_['book_mode_rating'].fillna(global_avg_rating)
    books_['book_rating_cv'] = books_['book_rating_cv'].fillna(0)
    # =================================================================================================

    # age 정보 전처리 ===================================================================================
    def age_range(age: float) -> str:
        if pd.isna(age):
            return 'unknown'
        
        age = int(age)

        if age <= 6:
            human = '유아기' 
        elif age <= 13:
            human = '아동기'
        elif age <= 19:
            human = '청소년기'
        elif age <= 34:
            human = '청년'
        elif age <= 49:
            human = '장년/중년 초기'
        elif age <= 64:
            human = '중년 후기'
        elif age <= 74:
            human = '노년 초기'
        else:
            human = '노년 후기'
            
        return human

    # 10대, 20대, 30대, ... + 결측치는 최빈값으로 대체
    users_['age_for_10'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_10'] = users_['age_for_10'].apply(lambda x: x // 10 * 10) 

    # 인문학적 연령대 + 결측치 정보 포함
    users_['age_group'] = users_['age'].apply(lambda x: age_range(x))

    users_ = users_.drop(['age', 'age_for_10'], axis=1, errors='ignore')

    ratings_with_age = ratings_explicit.merge(
        users_[['user_id', 'age_group']],
        on='user_id',
        how='left'
    )

    age_group_avg_rating_df = ratings_with_age.groupby('age_group')['rating'].agg(
        age_group_avg_rating='mean'
    ).reset_index()

    users_ = users_.merge(
        age_group_avg_rating_df, 
        on='age_group', 
        how='left'
    )

    users_['age_group_avg_rating'] = users_['age_group_avg_rating'].fillna(global_avg_rating)
    # =================================================================================================

    # author 정보 전처리 ==============================================================================
    author_stats_df = ratings_explicit.merge(books_[['isbn', 'book_author']], on='isbn', how='left')
    author_stats_df = author_stats_df.groupby('book_author')['rating'].agg(
        author_avg_rating='mean',      # 저자 평균 평점 (연속형)
        author_book_count='count',      # 저자 평점 횟수 (활동성)
        author_rating_std='std'
    ).reset_index()

    books_ = books_.merge(
        author_stats_df[['book_author', 'author_avg_rating', 'author_book_count', 'author_rating_std']],
        on='book_author',
        how='left'
    )
    books_['author_avg_rating'] = books_['author_avg_rating'].fillna(global_avg_rating)
    books_['author_rating_std'] = books_['author_rating_std'].fillna(global_std_rating)
    books_.loc[books_['author_book_count'].isna(), 'author_book_count'] = 0 # 저자 활동성 결측치는 0으로

    def categorize_author_activity(count):
        if count <= 5: return 'A_RARE'
        elif count <= 20: return 'A_MID'
        else: return 'A_ACTIVE'
    
    books_['book_author'] = books_['book_author'].fillna('Unknown_Author')
    books_['author_activity_group'] = books_['author_book_count'].apply(categorize_author_activity)
    books_ = books_.drop(['author_book_count'], axis=1)
    # =================================================================================================

    # publisher 정보 전처리 ==============================================================================
    MIN_BOOKS_PUBLISHER = 90      # 출판사 90개 미만
    books_['publisher'] = books_['publisher'].apply(lambda x: normalize_category_element(x))
    books_ = handle_rare(books_, 'publisher', min_count=MIN_BOOKS_PUBLISHER, replacement_name='RARE_PUBLISHER')
    
    books_['publisher'] = books_['publisher'].fillna('UNKNOWN_PUBLISHER')

    publisher_stats_df = ratings_explicit.merge(books_[['isbn', 'publisher']], on='isbn', how='left')

    # publisher별 통계 계산
    publisher_stats_df = publisher_stats_df.groupby('publisher')['rating'].agg(
        publisher_avg_rating='mean',   # 출판사 평균 평점
        publisher_book_count='count',   # 출판사 평점 횟수 (책의 규모 대신 평점 받은 횟수 사용이 더 정확함)
        publisher_rating_std='std'      # 출판사 분산
    ).reset_index()

    # books_에 병합하고 결측값 처리 (결측값은 global_avg_rating 또는 0으로 처리)
    books_ = books_.merge(
        publisher_stats_df,
        on='publisher',
        how='left'
    )
    
    books_['publisher_avg_rating'] = books_['publisher_avg_rating'].fillna(global_avg_rating)
    books_['publisher_rating_std'] = books_['publisher_rating_std'].fillna(global_std_rating)
    books_['publisher_book_count'] = books_['publisher_book_count'].fillna(0)
    # =================================================================================================

    # category 정보 전처리 ==============================================================================
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['category'] = books_['category'].apply(lambda x: normalize_category_element(x))

    MIN_BOOKS_CATEGORY = 20      # 카테고리 20개 미만
    books_ = handle_rare(books_, 'category', min_count=MIN_BOOKS_CATEGORY, replacement_name='RARE_CATEGORY')

    books_['category'] = books_['category'].fillna('UNKNOWN_CATEGORY')

    ratings_with_category = ratings_explicit.merge(
        books_[['isbn', 'category']],
        on='isbn',
        how='left'
    )
    
    category_avg_rating_df = ratings_with_category.groupby('category')['rating'].agg(
        category_avg_rating='mean'
    ).reset_index()

    books_ = books_.merge(
        category_avg_rating_df, 
        on='category', 
        how='left'
    )

    books_['category_avg_rating'] = books_['category_avg_rating'].fillna(global_avg_rating)
    # =================================================================================================

    # rating 구간화 (Stratified K-Fold) ================================================================
    def create_rating_bins(rating):
        """평점을 5개의 구간(bin)으로 나누는 함수 (1-10점 기준)"""
        if rating <= 2:
            return 0
        elif rating <= 4:
            return 1
        elif rating <= 6:
            return 2
        elif rating <= 8:
            return 3
        else: # 9, 10점
            return 4

    ratings_explicit['rating_bin'] = ratings_explicit['rating'].apply(create_rating_bins)
    # =================================================================================================

    return users_, books_


def context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_context_data(users, books, train)
    

    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.

    # USER 범주형 피처 (Sparse, Embedding 필요)
    # 추가 가능한 feature:
    sparse_user_features = ['user_id',
                            'age_group', 'age_10',
                            'location_country', 'location_city', 'location_state',
                            'user_activity_level']
    
    # BOOK 범주형 피처 (Sparse, Embedding 필요)
    # 추가 가능한 feature: 
    sparse_book_features = ['isbn',
                            'book_author','publisher', 'category', 'language', 'has_summary',
                            'publication_range', 'author_activity_group', 'book_rarity_level']
    
    # 연속형 피처 (Continuous, DNN에 직접 입력)
    # 추가 가능한 feature: 'book_title_length', 'user_mode_rating', 'user_avg_rating', 'user_rating_std', 'book_rating_std'
    # 'book_mode_rating', 'book_rating_skewness', 'book_rating_kurtosis', 'book_avg_rating', 'author_rating_std', 'publisher_rating_std'
    continuous_features = [
        # === [핵심 편향 및 활동성 (User)] ===
        # 'user_global_bias',          # ★★★ user_avg_rating 대체 (핵심 예측력 0.54)
        'user_rating_cv',            # 사용자 평점 변동성 (Std 제거)
        'user_rating_count',         # 사용자 활동 빈도
        'age_group_avg_rating',      # 집단 편향 (신규 추가 피처)
        
        # === [핵심 편향 및 활동성 (Book)] ===
        # 'book_global_bias',          # ★★★ book_avg_rating 대체 (핵심 예측력 0.71)
        'book_rating_cv',            # 책 평점 변동성 (Std 제거)
        'book_rating_count',         # 책 인기도
        
        # === [저자/출판사 통계] ===
        'author_avg_rating',         # 저자 평균 평점 (Std 제거)
        'publisher_avg_rating',      # 출판사 평균 평점 (Std 제거)
        'publisher_book_count',      # 출판사 출판 책 수
        
        # === [기타 통계 (약한 상관관계 피처)] ===
        'user_rating_skewness', 'user_rating_kurtosis',
        'book_rating_skewness', 'book_rating_kurtosis',
        
        # === [기타 도서 메타데이터] ===
        'book_age', 'book_title_word_count', 'category_avg_rating'
    ]

    # sparse_cols 정의 (ID 및 범주형 피처)
    # NCF는 user_id, isbn이 필수이므로 모델에 따라 분리 로직을 적용
    all_sparse_features = list(set(sparse_user_features + sparse_book_features))

    if args.model == 'NCF':
        sparse_cols = ['user_id', 'isbn'] + [col for col in all_sparse_features if col not in ['user_id', 'isbn']]
    else:
        sparse_cols = all_sparse_features
    
    all_cols = sparse_cols + continuous_features
    text_embedding_cols = ['user_summary_merge_vector', 'book_summary_vector']

    # 1단계: 임베딩 컬럼까지 포함된 전체 DataFrame을 먼저 생성
    # 이때는 all_cols에 없는 임베딩 컬럼도 users_와 books_에서 모두 가져와 포함합니다.
    full_train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    full_test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')

    # 2단계: 필요한 Text Embedding 컬럼을 분리하여 별도로 저장
    train_text_data = full_train_df[text_embedding_cols].copy()
    test_text_data = full_test_df[text_embedding_cols].copy()

    # 3단계: train_df와 test_df를 필요한 컬럼(sparse + continuous + rating)으로만 제한
    # 이 시점에서 Text Embedding 컬럼은 drop 됩니다.
    train_df = full_train_df[all_cols + ['rating']].copy()
    test_df = full_test_df[all_cols].copy()
    
    train_df['country_x_language'] = train_df['location_country'].astype(str) + '_' + train_df['language'].astype(str)
    test_df['country_x_language'] = test_df['location_country'].astype(str) + '_' + test_df['language'].astype(str)

    train_df['activity_rarity_level'] = (train_df['user_activity_level'].astype(str) + '_' + train_df['book_rarity_level'].astype(str))
    test_df['activity_rarity_level'] = (test_df['user_activity_level'].astype(str) + '_' + test_df['book_rarity_level'].astype(str))
    
    train_df['age_category_pref'] = (train_df['age_group'].astype(str) + '_' + train_df['category'].astype(str))
    test_df['age_category_pref'] = (test_df['age_group'].astype(str) + '_' + test_df['category'].astype(str))
    
    train_df['country_publisher_link'] = (train_df['location_country'].astype(str) + '_' + train_df['publisher'].astype(str))
    test_df['country_publisher_link'] = (test_df['location_country'].astype(str) + '_' + test_df['publisher'].astype(str))
    
    train_df['user_author_activity'] = (train_df['user_activity_level'].astype(str) + '_' + train_df['author_activity_group'].astype(str))
    test_df['user_author_activity'] = (test_df['user_activity_level'].astype(str) + '_' + test_df['author_activity_group'].astype(str))

    # 추가 가능한 feature:
    new_sparse_cols = ['country_x_language', 'user_author_activity', 'age_category_pref', 'activity_rarity_level', 'country_publisher_link']
                       
    for col in new_sparse_cols:
        if col not in sparse_cols:
            sparse_cols.append(col)

    train_df = train_df[sparse_cols + continuous_features + ['rating']]
    test_df = test_df[sparse_cols + continuous_features]

    all_df = pd.concat([train_df, test_df], axis=0)

    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        cat_type = all_df[col].astype("category") 
        unique_labels = cat_type.cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=cat_type.cat.categories).codes
        test_df[col] = pd.Categorical(test_df[col], categories=cat_type.cat.categories).codes
        if (train_df[col] < 0).any():
            print(f"경고: {col} 컬럼에서 Train 데이터에 -1 인덱스 발견. 0으로 대체.")
            train_df.loc[train_df[col] < 0, col] = 0
            
        if (test_df[col] < 0).any():
            print(f"경고: {col} 컬럼에서 Test 데이터에 -1 인덱스 발견. 0으로 대체.")
            test_df.loc[test_df[col] < 0, col] = 0

    scaler = StandardScaler()

    train_continuous_data = train_df[continuous_features].fillna(train_df[continuous_features].mean())
    scaler.fit(train_continuous_data) 

    test_continuous_data = test_df[continuous_features].fillna(train_df[continuous_features].mean())
    train_df[continuous_features] = scaler.transform(train_continuous_data)
    test_df[continuous_features] = scaler.transform(test_continuous_data)

    field_dims = [len(label2idx[col]) for col in train_df.columns if col in sparse_cols]
    
    data = {
            'train':train_df,
            'test':test_df,
            'sparse_field_names':sparse_cols,
            'continuous_field_names':continuous_features,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'train_text_data': train_text_data,
            'test_text_data': test_text_data,
            'text_embedding_cols': text_embedding_cols,
            'sub':sub,
            }

    return data


def context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)


def context_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    sparse_cols = data['sparse_field_names']
    continuous_features = data['continuous_field_names']
    text_embedding_cols = data['text_embedding_cols']

    def process_text_embedding(text_df, text_cols, args):
        current_model = args.model
        embed_dim = args.model_args[current_model]['text_embed_dim']
        zero_vector = [0.0] * embed_dim

        processed_df = text_df[text_cols].copy()

        for col in text_cols:
            processed_df[col] = processed_df[col].apply(
                lambda vec: np.array(vec, dtype=np.float32) 
                            # vec이 리스트나 NumPy 배열 타입이며, 길이가 0보다 클 때만 배열로 변환
                            if isinstance(vec, (list, np.ndarray)) and len(vec) > 0
                            else np.array(zero_vector, dtype=np.float32)
            )
        
        combined_vectors = processed_df.apply(lambda row: np.concatenate(row.values), axis=1)

        final_array = np.stack(combined_vectors.values)
        
        return torch.FloatTensor(final_array)

    # Train Text
    X_train_text = process_text_embedding(data['train_text_data'].loc[data['X_train'].index], text_embedding_cols, args)
    
    # Valid Text
    if args.dataset.valid_ratio != 0:
        X_valid_text = process_text_embedding(data['train_text_data'].loc[data['X_valid'].index], text_embedding_cols, args)
    else:
        X_valid_text = None
        
    # Test Text
    X_test_text = process_text_embedding(data['test_text_data'], text_embedding_cols, args)

    # Train
    X_train_sparse = torch.LongTensor(data['X_train'][sparse_cols].values)
    X_train_continuous = torch.FloatTensor(data['X_train'][continuous_features].values)
    y_train = torch.FloatTensor(data['y_train'].values)
    
    train_dataset = TensorDataset(X_train_sparse, X_train_continuous, X_train_text, y_train)

    # Valid
    if args.dataset.valid_ratio != 0:
        X_valid_sparse = torch.LongTensor(data['X_valid'][sparse_cols].values)
        X_valid_continuous = torch.FloatTensor(data['X_valid'][continuous_features].values)
        y_valid = torch.FloatTensor(data['y_valid'].values)
        valid_dataset = TensorDataset(X_valid_sparse, X_valid_continuous, X_valid_text, y_valid)
    else:
        valid_dataset = None

    # Test
    X_test_sparse = torch.LongTensor(data['test'][sparse_cols].values)
    X_test_continuous = torch.FloatTensor(data['test'][continuous_features].values)
    
    test_dataset = TensorDataset(X_test_sparse, X_test_continuous, X_test_text)

    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if valid_dataset is not None else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
