Book Recommendation System
   Multi-Modal Fusion
========================================================================

1. 프로젝트 개요 (OVERVIEW)
----------------------------
이 프로젝트는 도서 평점 예측(Rating Prediction)을 위한 하이브리드 추천 시스템입니다.
전통적인 협업 필터링 요소(ID 및 특징)에 CLIP 기반의 이미지/텍스트 다중 양식(Multi-modal) 임베딩과 아이템의 통계적 특성(빈도, 평균/표준편차)을 융합하여 예측 성능과 견고성을 높이는 것을 목표로 합니다.

2. 기술 스택 (TECHNOLOGY STACK)
-------------------------------
* 언어: Python 3.8+
* 프레임워크: PyTorch, scikit-learn
* 데이터 처리: Pandas, NumPy
* 하드웨어: MPS

3. 데이터셋 구조 (bookset Class)
--------------------------------
데이터셋 클래스 'bookset'은 전처리가 완료된 평점 데이터를 PyTorch의 Dataset 형식으로 제공합니다.
모든 ID는 [0, N-1] 범위의 정수형으로 인코딩되어야 합니다.

[주요 특징 및 데이터 타입]
- user (user_id): 사용자 고유 ID
- user_f0 (location): 사용자 지역 ID
- user_f1 (age): 사용자 연령 ID
- item (isbn): 책 고유 ID
- item_f0 (book_author): 저자 ID
- item_f1 (year_of_publication): 출판 연도 ID
- item_f2 (publisher): 출판사 ID
- item_f3 (language): 언어 ID
- item_f4 (category): 카테고리 ID
- y (rating): 정답 레이블 (평점, torch.float)

4. 모델 아키텍처 (Model Class)
------------------------------
모델은 9개의 임베딩 테이블과 다중 양식 데이터를 통합합니다.

[핵심 메커니즘]
A. 임베딩 통합 (Embedding Fusion):
    9가지 특징(유저 3개 + 아이템 6개) 임베딩을 결합합니다.

B. Frequency Gating:
    학습 가능한 가중치(w)와 아이템 빈도(L = log(freq))를 사용하여 9개 임베딩의 가중치(alpha)를 계산합니다. 이는 저빈도 아이템 예측 시 특징의 중요도를 동적으로 조절합니다.
    - alpha = sigmoid(L * w)

C. Multi-modal Integration:
    CLIP 이미지 및 텍스트 임베딩을 선형 투영(Projection)한 후, 특징 가중치(w)와 상호 보완적으로 작용하도록 스케일링하여 최종 예측 벡터에 통합합니다.

D. 통계 정보 결합:
    아이템별 평점 평균 및 표준편차 통계량을 최종 예측 벡터에 직접 연결합니다.

E. 최종 예측:
    통합된 벡터를 Multi-Layer Perceptron (MLP)에 입력하여 최종 평점을 예측합니다.

5. 실행 예시 (USAGE EXAMPLE)
--------------------------
# 1. Dataset 및 DataLoader 생성 (분리된 X_train, y_train을 입력)
# train_data = bookset(X_train, y_train, device='mps')
# train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# 2. 모델 인스턴스화
# model = Model(user_cnt=..., item_cnt=..., ..., embedding_dim=64, freq_of_item=..., device='mps')
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()

# 3. 학습 루프 (Training Loop)
# for features, ratings in train_loader:
#     # features는 bookset의 __getitem__ 순서대로 언팩
#     outputs = model(*features)
#     loss = criterion(outputs, ratings)
#     loss.backward()
#     optimizer.step()
