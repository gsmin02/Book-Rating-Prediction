# Book Rating Prediction (RecSys)

## 1. Project Overview
본 프로젝트는 사용자–아이템 상호작용과 메타데이터를 활용하여
도서 평점을 예측하는 추천 시스템을 구축하는 것을 목표로 한다.

최종 모델은 XGBoost, LightGBM를 결합한
가중치 기반 앙상블 모델이다.

## 2. Dataset
- Users
- Books
- Ratings

## 3. Models
- XGBoost (tree-based regression)
- LightGBM (gradient boosting)

## 4. Ensemble
- Weighted ensemble based on validation RMSE

## 5. How to Run
1. notebooks/ 폴더의 노트북을 순서대로 실행
2. 최종 submission 파일 생성

## 6. Environment
- Python 3.10
- XGBoost
- LightGBM
- PyTorch
