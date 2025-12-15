import wandb
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from label_encoding import label_encoding
from create_embedding import load_or_create_img_embedding, load_or_create_txt_embedding, load_or_create_embedding
from model_utils import get_lgb_params, save_prediction, train_lgb_model
from feature_engineering import add_user_book_features


def import_csv():
    #데이터 불러오기
    data_path = "/data/ephemeral/home/code/data"

    books_df = pd.read_csv(data_path + "/books.csv")
    user_df = pd.read_csv(data_path + "/users.csv")
    train_df = pd.read_csv(data_path + "/train_ratings.csv")
    test_df = pd.read_csv(data_path + "/test_ratings.csv")

    #train_df, test_df에 각각 books_df, user_df를 user_id, isbn 기준 inner join
    merged_train_df = pd.merge(train_df, user_df, on="user_id", how="inner")
    merged_train_df = pd.merge(merged_train_df, books_df, on="isbn", how="inner")

    merged_test_df = pd.merge(test_df, user_df, on="user_id", how="inner")
    merged_test_df = pd.merge(merged_test_df, books_df, on="isbn", how="inner")

    #train과 test 합침(일단 모든 데이터에 대해 임베딩을 만들기 위해서)
    full_df = pd.concat([merged_train_df, merged_test_df], axis=0, ignore_index=True)
    train_rows = len(merged_train_df)
    return full_df, test_df,train_rows

def load_data():
    id_cols=["user_id", "isbn", "category"]
    text_cols=["summary", "book_title", "book_author", "publisher", "location", "language"]

    full_df, test_df, train_rows = import_csv()
    full_df = add_user_book_features(full_df, train_rows)

    # ========= ID 인코딩 =========
    label_encoding(id_cols, full_df)

    # ========= 숫자 feature =========
    num_cols = ["age", "year_of_publication", "book_count", "user_count","book_mean_rating","user_mean_rating", "book_std_rating", "user_std_rating","author_mean_rating","publisher_mean_rating"]
    num_features = full_df[num_cols].fillna(0).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ========= 이미지 임베딩 =========
    image_embeddings = load_or_create_img_embedding(
        "embeddings/image_embeddings.npy",
        full_df,
        device
    )
    image_col_names = [f"image_emb_{i}" for i in range(image_embeddings.shape[1])]
    df_image = pd.DataFrame(image_embeddings, columns=image_col_names)

    # ========= 텍스트 임베딩 =========
    print("텍스트 임베딩 로딩 시작...")

    txt_emb_dict = load_or_create_txt_embedding(
        "embeddings/text_embedding",
        full_df,
        text_cols,
        device
    )

    text_feature_blocks = []
    text_feature_names = []

    for key, emb in txt_emb_dict.items():
        # emb: (N, dim)
        col_names = [f"{key}_emb_{i}" for i in range(emb.shape[1])]
        text_feature_blocks.append(
            pd.DataFrame(emb, columns=col_names)
        )
        text_feature_names.extend(col_names)

    # 전체 텍스트 feature DataFrame
    if len(text_feature_blocks) > 0:
        df_text_all = pd.concat(text_feature_blocks, axis=1)
    else:
        df_text_all = pd.DataFrame(index=range(len(full_df)))

    print("텍스트 feature 전체 shape:", df_text_all.shape)

    # ========= Feature merge =========
    X_full = pd.concat(
        [
            full_df[id_cols].reset_index(drop=True),
            pd.DataFrame(num_features, columns=num_cols),
            df_text_all.reset_index(drop=True),
            df_image.reset_index(drop=True)
        ],
        axis=1
    )

    X_train_full = X_full.iloc[:train_rows].copy()
    X_test = X_full.iloc[train_rows:].copy()
    y_train_full = full_df["rating"].values[:train_rows]

    return X_train_full, y_train_full, X_test, test_df

    