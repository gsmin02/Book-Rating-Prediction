import pandas as pd
import numpy as np

def add_user_book_features(full_df, train_rows):
    #train/test 분리
    train_df = full_df.iloc[:train_rows].copy()

    #train에 있는 rating 데이터만 이용해서 isbn, user_id 별 평점 평균과 분산, 평점 개수, 작가와 출판사 별 평점 평균 계산
    isbn_count = train_df.groupby("isbn").size().reset_index(name="book_count")
    user_count = train_df.groupby("user_id").size().reset_index(name="user_count")

    book_mean = train_df.groupby("isbn")["rating"].mean().reset_index(name="book_mean_rating")
    user_mean = train_df.groupby("user_id")["rating"].mean().reset_index(name="user_mean_rating")

    book_std = train_df.groupby("isbn")["rating"].std().reset_index(name="book_std_rating")
    user_std = train_df.groupby("user_id")["rating"].std().reset_index(name="user_std_rating")

    author_mean = train_df.groupby("book_author")["rating"].mean().reset_index(name="author_mean_rating")
    publisher_mean = train_df.groupby("publisher")["rating"].mean().reset_index(name="publisher_mean_rating")

    #전체 평균(결측치를 채우기 위해)
    global_book_mean = book_mean["book_mean_rating"].mean()
    global_user_mean = user_mean["user_mean_rating"].mean()
    global_book_std = book_std["book_std_rating"].mean()
    global_user_std = user_std["user_std_rating"].mean()
    global_book_count = isbn_count["book_count"].mean()
    global_user_count = user_count["user_count"].mean()
    global_author_mean = author_mean["author_mean_rating"].mean()
    global_publisher_mean = publisher_mean["publisher_mean_rating"].mean()

    #full_df에 merge
    full_df = pd.merge(full_df, isbn_count, on="isbn", how="left")
    full_df = pd.merge(full_df, user_count, on="user_id", how="left")
    full_df = pd.merge(full_df, book_mean, on="isbn", how="left")
    full_df = pd.merge(full_df, user_mean, on="user_id", how="left")
    full_df = pd.merge(full_df, book_std, on="isbn", how="left")
    full_df = pd.merge(full_df, user_std, on="user_id", how="left")
    full_df = pd.merge(full_df, author_mean, on="book_author", how="left")
    full_df = pd.merge(full_df, publisher_mean, on="publisher", how="left")

    #train에 존재하는 isbn/user 리스트
    train_isbns = set(train_df["isbn"].unique())
    train_users = set(train_df["user_id"].unique())

    #train 데이터에 없는 user가 test에 나타난 경우
    #   train데이터에 헤당 행의 isbn이 있으면 isbn별 평균과 표준편차로 채움
    #반대의 경우도 마찬가지
    #user, isbn 둘 다 train에 없으면 전체 평균으로 채움 
    def conditional_fill(row, col, is_isbn=True):
        value = row[col]
        if pd.notna(value) and value != 0:
            return value

        if is_isbn:
            if row["isbn"] in train_isbns:
                return row[col]  
            else:
                return None 
        else:
            if row["user_id"] in train_users:
                return row[col]
            else:
                return None

    for col, is_isbn in [
        ("book_mean_rating", True),
        ("book_std_rating", True),
        ("book_count", True),
        ("user_mean_rating", False),
        ("user_std_rating", False),
        ("user_count", False),
    ]:
        full_df[col] = full_df.apply(lambda r: conditional_fill(r, col, is_isbn), axis=1)

    #남아있는 NaN을 전체 평균으로 채움
    full_df["book_mean_rating"].fillna(global_book_mean, inplace=True)
    full_df["user_mean_rating"].fillna(global_user_mean, inplace=True)
    full_df["book_std_rating"].fillna(global_book_std, inplace=True)
    full_df["user_std_rating"].fillna(global_user_std, inplace=True)
    full_df["book_count"].fillna(global_book_count, inplace=True)
    full_df["user_count"].fillna(global_user_count, inplace=True)
    full_df["author_mean_rating"].fillna(global_author_mean, inplace=True)
    full_df["publisher_mean_rating"].fillna(global_publisher_mean, inplace=True)

    return full_df
