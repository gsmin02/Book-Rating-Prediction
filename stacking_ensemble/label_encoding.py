from sklearn.preprocessing import LabelEncoder

def label_encoding(id_cols, full_df):
    id_encoders = {}

    for col in id_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))
        id_encoders[col] = le
    print("Label Encoding:,",full_df[id_cols].values.shape, full_df[id_cols].values.dtype, type(full_df[id_cols].values))
    print("Label Encoding 완료")
    return id_encoders