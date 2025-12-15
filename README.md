# 파일 구조
stacking_ensemble/
├── train.py        #전체 파이프라인 실행 스크립트
├── get_oof_prediction.py       #각 모델별 OOF 예측 함수
├── oof_models.py               #base model factory 함수
├── ncf_wrapper.py              #NCF 모델 wrapper
├── deepfm_wrapper.py           #DeepFM 모델 wrapper
├── load_data.py                #LightGBM용 데이터 로드
├── load_ncf_data.py            #NCF/DeepFM용 context 데이터 처리
├── models/
│   ├── NCF.py                  #Neural Collaborative Filtering 모델 정의
│   └── DeepFM.py               #DeepFM 모델 정의
├── output/
│   └── result.csv              #최종 제출물
└── README.md
