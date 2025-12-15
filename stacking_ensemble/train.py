from sklearn.model_selection import KFold
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import subprocess
from stacking_ensemble.models.oof_models import lgb_oof_model, create_ncf_model, create_deepfm_model
from load_data.load_data import import_csv, load_data
from get_oof_prediction import get_xgb_oof_predictions,get_lgb_oof_predictions, get_ncf_oof_predictions, get_deepfm_oof_predictions,get_rf_oof_predictions
from wrapper.ncf_wrapper import NCFWrapper
from load_data.load_ncf_data import context_data_load, context_data_loader
from sklearn.metrics import mean_squared_error

#데이터 불러오기
lgb_X_train, lgb_y_train, lgb_X_test, lgb_test_df=load_data()

context_data = context_data_load()
context_data = context_data_loader(context_data)

sparse_cols = context_data['field_names']  
field_dims = [len(context_data['label2idx'][col]) for col in sparse_cols]
context_X_train = context_data['X_train'].values
context_y_train = context_data['y_train'].values
context_X_test  = context_data['test'].values

#베이스 모델
lgb_model = lgb_oof_model(0.1)
lgb_model2 = lgb_oof_model(0.2)
lgb_model3 = lgb_oof_model(0.05)

# OOF 예측 생성
#rf_oof, rf_test = get_rf_oof_predictions(lgb_X_train, lgb_y_train, lgb_X_test)
#xgb_oof, xgb_test = get_xgb_oof_predictions(lgb_X_train, lgb_y_train, lgb_X_test)
#deepfm_oof, deepfm_test = get_deepfm_oof_predictions(field_dims,create_deepfm_model, context_X_train, context_y_train, context_X_test)
#ncf_oof, ncf_test = get_ncf_oof_predictions(field_dims,create_ncf_model, context_X_train,context_y_train,context_X_test)
lgb_oof, lgb_test = get_lgb_oof_predictions(lgb_model, lgb_X_train, lgb_y_train, lgb_X_test)
lgb_oof2, lgb_test2 = get_lgb_oof_predictions(lgb_model2, lgb_X_train, lgb_y_train, lgb_X_test)
# ========== Meta Model ==========
stack_train = np.vstack([
    #deepfm_oof,
    lgb_oof,
    #ncf_oof,
    lgb_oof2          
]).T

stack_test = np.vstack([
    #deepfm_test,
    lgb_test,
    #ncf_test,
    lgb_test2         
]).T

meta_model = LinearRegression()
meta_model.fit(stack_train, lgb_y_train)

final_pred = meta_model.predict(stack_test)
final_pred = np.clip(final_pred, 1, 10)

val_rmse = np.sqrt(mean_squared_error(lgb_y_train, np.clip(meta_model.predict(stack_train), 1, 10)))
print("Meta model OOF RMSE (validation):", val_rmse)
print("Meta model weights (coefficients):")
print(meta_model.coef_)

lgb_test_df['rating']=final_pred
output_path = f"output/result.csv"
lgb_test_df.to_csv(output_path, index=False)
print("저장 완료:", output_path)
