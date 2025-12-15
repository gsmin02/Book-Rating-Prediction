import pandas as pd
import lightgbm as lgb
import torch
import numpy as np
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary

def get_lgb_params(config):
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "verbose": -1,
        "max_depth": -1
    }

    return base_params


def train_lgb_model(params, X_train, y_train, X_valid, y_valid, id_cols):
    # wandb.init() 절대 쓰지 말 것
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Feature Importance 기록
    fi = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance()
    })
    wandb.log({"feature_importance": wandb.Table(dataframe=fi)})

    return model


def save_prediction(model, X_test, test_df, best_score, lr):
    """예측 + 클리핑 + CSV 저장 기능만 담당"""
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 10)


    test_df["rating"] = y_pred

    run_id = wandb.run.id
    output_path = f"result_csv/result_{run_id}_{best_score:.4f}.csv"
    test_df.to_csv(output_path, index=False)
    print("저장 완료:", output_path)
    return output_path
