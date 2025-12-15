"""
main_enhanced.py
Enhanced DeepFM 모델 실행 스크립트

사용법:
    python main_enhanced.py --config config/config_enhanced.yaml

또는 직접 실행:
    python main_enhanced.py
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # ============ 설정 ============
    class Args:
        """간단한 설정 클래스"""
        # 기본 설정
        seed = 42
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = 'DeepFM_Enhanced'
        
        # 데이터 설정
        class dataset:
            data_path = 'data/'
            valid_ratio = 0.2
        
        # 데이터로더 설정
        class dataloader:
            batch_size = 1024
            shuffle = True
            num_workers = 0
        
        # 모델 설정
        class model_args:
            embed_dim = 16
            mlp_dims = [64, 32, 16]
            batchnorm = True
            dropout = 0.3
        
        # 옵티마이저 설정
        class optimizer:
            class args:
                lr = 1e-3
                weight_decay = 1e-4
        
        # 학습 설정
        class train:
            epochs = 30
            ckpt_dir = 'saved/checkpoint'
            submit_dir = 'saved/submit'
            save_best_model = True
        
        # LR Scheduler 설정
        class lr_scheduler:
            use = True
        
        # Enhanced 설정
        class enhanced:
            apply_cold_postprocess = True
            cold_weight = 0.5
    
    args = Args()
    
    # ============ 시드 고정 ============
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    # ============ 디렉토리 생성 ============
    os.makedirs(args.train.ckpt_dir, exist_ok=True)
    os.makedirs(args.train.submit_dir, exist_ok=True)
    
    # ============ 데이터 로드 ============
    print("\n" + "="*50)
    print("Loading Enhanced Context Data...")
    print("="*50)
    
    from enhanced_context_data import (
        enhanced_context_data_load,
        enhanced_context_data_split,
        enhanced_context_data_loader
    )
    
    data = enhanced_context_data_load(args)
    print(f"Train size: {len(data['train']):,}")
    print(f"Test size: {len(data['test']):,}")
    print(f"Sparse features: {data['field_names']}")
    print(f"Dense features: {data['dense_cols']}")
    print(f"Field dims: {data['field_dims']}")
    
    data = enhanced_context_data_split(args, data)
    print(f"X_train: {len(data['X_train']):,}, X_valid: {len(data['X_valid']):,}")
    
    data = enhanced_context_data_loader(args, data)
    print("DataLoader created!")
    
    # ============ 모델 생성 ============
    print("\n" + "="*50)
    print(f"Initializing {args.model}...")
    print("="*50)
    
    from DeepFM_Enhanced import DeepFM_Enhanced
    
    # 모델에 전달할 데이터 정보
    model_data = {
        'field_dims': data['field_dims'],
        'dense_dim': data['dense_dim'],
        'stats': data['stats'],
    }
    
    model = DeepFM_Enhanced(args.model_args, model_data)
    model = model.to(args.device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============ 학습 ============
    print("\n" + "="*50)
    print("Training...")
    print("="*50)
    
    from train_enhanced import train_enhanced, test_enhanced
    
    model = train_enhanced(args, model, data)
    
    # ============ 예측 ============
    print("\n" + "="*50)
    print("Predicting...")
    print("="*50)
    
    predictions = test_enhanced(
        args, model, data,
        apply_cold_postprocess=args.enhanced.apply_cold_postprocess,
        cold_weight=args.enhanced.cold_weight
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Predictions mean: {predictions.mean():.2f}")
    
    # ============ 제출 파일 생성 ============
    print("\n" + "="*50)
    print("Saving submission...")
    print("="*50)
    
    submission = data['sub'].copy()
    submission['rating'] = predictions
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.train.submit_dir}/{timestamp}_{args.model}.csv"
    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    
    # ============ 요약 ============
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Sparse features: {len(data['field_names'])}")
    print(f"Dense features: {data['dense_dim']}")
    print(f"Cold Start postprocess: {args.enhanced.apply_cold_postprocess}")
    print(f"Submission: {filename}")
    
    return predictions


if __name__ == "__main__":
    predictions = main()
