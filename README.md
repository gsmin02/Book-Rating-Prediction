# 파일 구조
stacking_ensemble/
├── stacking_ensemble.py # 전체 스태킹 앙상블 파이프라인 실행 스크립트
├── get_oof_prediction.py # 모델별 OOF 예측 함수 모음
├── oof_models.py # Base model 생성(factory) 함수
├── ncf_wrapper.py # NCF 학습/예측을 위한 wrapper 클래스
├── deepfm_wrapper.py # DeepFM 학습/예측을 위한 wrapper 클래스
├── load_data.py # LightGBM용 tabular 데이터 로드
├── load_ncf_data.py # NCF/DeepFM용 context 데이터 전처리 및 로더
├── models/
│ ├── NCF.py # Neural Collaborative Filtering 모델 정의
│ └── DeepFM.py # DeepFM 모델 정의
├── output/
│ └── result.csv # 최종 예측 결과 (clipping 적용)
└── README.md # 프로젝트 설명 문서
