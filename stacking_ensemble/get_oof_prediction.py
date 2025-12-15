from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_lgb_oof_predictions(model, X, y, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_splits

    oof = np.clip(oof, 1, 10)
    test_mean = np.clip(test_mean, 1, 10)
    return oof, test_preds

def get_ncf_oof_predictions(field_dims,model_factory, X, y, X_test, n_splits=5, verbose=True):
    """
    model_factory: fold마다 새로운 모델 인스턴스를 반환하는 함수
    X, X_test: numpy array
    y: numpy array
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = np.asarray(X)
    X_test = np.asarray(X_test)
    y = np.asarray(y)

    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_splits), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"Fold {fold+1}/{n_splits}")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # fold마다 새 모델 생성
        model = model_factory(field_dims)
        model.fit(X_tr, y_tr, X_valid=X_val, y_valid=y_val, verbose=verbose)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof[val_idx] = val_pred
        test_preds[:, fold] = test_pred

    # test 예측은 fold별 평균
    test_mean = test_preds.mean(axis=1)
    oof = np.clip(oof, 1, 10)
    test_mean = np.clip(test_mean, 1, 10)
    return oof, test_mean

def get_deepfm_oof_predictions(field_dims,model_factory, X, y, X_test, n_splits=5, verbose=True):
    """
    DeepFM용 OOF 예측 생성
    model_factory: fold마다 새로운 모델 인스턴스를 반환하는 함수
    X, X_test: numpy array
    y: numpy array
    """
    X = np.asarray(X)
    X_test = np.asarray(X_test)
    y = np.asarray(y)

    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_splits), dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"Fold {fold+1}/{n_splits}")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # fold마다 새 모델 생성
        model = model_factory(field_dims)
        model.fit(X_tr, y_tr, X_valid=X_val, y_valid=y_val, verbose=verbose)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof[val_idx] = val_pred
        test_preds[:, fold] = test_pred

    # test 예측은 fold별 평균
    test_mean = test_preds.mean(axis=1)
    oof = np.clip(oof, 1, 10)
    test_mean = np.clip(test_mean, 1, 10)
    return oof, test_mean

def get_rf_oof_predictions(X, y, X_test, n_splits=5):
    # pandas DataFrame → numpy array 변환
    if hasattr(X, "values"):
        X = X.values
    if hasattr(X_test, "values"):
        X_test = X_test.values

    y = np.array(y)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros((n_splits, len(X_test)), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        oof[val_idx] = model.predict(X_val)
        test_preds[fold] = model.predict(X_test)

        print(f"[RF] Fold {fold+1}/{n_splits} 완료")

    test_mean = test_preds.mean(axis=0)
    oof = np.clip(oof, 1, 10)
    test_mean = np.clip(test_mean, 1, 10)
    return oof, test_mean

def get_xgb_oof_predictions(X_train, y_train, X_test, n_splits=5):
    X = np.array(X_train)
    y = np.array(y_train)
    test = np.array(X_test)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(X), dtype=np.float32)
    preds_test = np.zeros(len(test), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"XGB Fold {fold+1}/{n_splits}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        oof[val_idx] = model.predict(X_val)
        preds_test += model.predict(test) / n_splits
    oof = np.clip(oof, 1, 10)
    preds_test = np.clip(preds_test, 1, 10)
    return oof, preds_test