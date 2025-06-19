#!/usr/bin/env python3
"""
실제 학습 코드에서 데이터셋 로딩을 테스트하는 스크립트
"""
import os
import sys
import yaml
from collections import Counter
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

from cellvit.training.experiments.experiment_cell_classifier import ExperimentCellVitClassifier
from cellvit.training.base_ml.base_cli import ExperimentBaseParser

def custom_collate_fn(batch):
    """배치 내의 가변 크기 데이터를 처리하는 collate 함수"""
    images = []
    detections = []
    types = []
    names = []
    
    for item in batch:
        # numpy array를 tensor로 변환
        if isinstance(item[0], np.ndarray):
            images.append(torch.from_numpy(item[0]).float())
        else:
            images.append(item[0])
        detections.append(item[1])
        types.append(item[2])
        names.append(item[3])
    
    # 이미지만 스택으로 만들고 나머지는 리스트로 유지
    images = torch.stack(images)
    
    return images, detections, types, names

def test_dataset_loading():
    # 기본 설정 파일 로드
    config_path = "./Dataset_Old/train_configs/ViT256/fold_0.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 실험 설정
    experiment = ExperimentCellVitClassifier(default_conf=config)
    experiment.run_conf = copy.deepcopy(config)  # run_conf 직접 설정
    
    try:
        # 데이터셋 생성
        train_filelist = os.path.join("Dataset_Old", "splits", "fold_0", "train.csv")
        val_filelist = os.path.join("Dataset_Old", "splits", "fold_0", "val.csv")
        
        train_dataset, val_dataset = experiment.get_datasets(
            dataset="DetectionDataset",
            train_filelist=train_filelist,
            val_filelist=val_filelist,
            normalize_stains_train=False,
            normalize_stains_val=False
        )
        
        print("\n=== 데이터셋 기본 정보 ===")
        print(f"Train 데이터셋 크기: {len(train_dataset)}")
        print(f"Val 데이터셋 크기: {len(val_dataset)}")
        
        # 단일 샘플 확인
        print("\n=== 단일 샘플 구조 확인 ===")
        sample = train_dataset[0]
        print(f"이미지 타입: {type(sample[0])}, 형태: {sample[0].shape}")
        print(f"검출 개수: {len(sample[1])}")
        print(f"타입 개수: {len(sample[2])}")
        print(f"이미지 이름: {sample[3]}")
        
        # DataLoader 테스트 (커스텀 collate_fn 사용)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        print("\n=== DataLoader 테스트 ===")
        print("Train loader 배치 추출 테스트...")
        images, detections, types, names = next(iter(train_loader))
        print(f"배치 구조:")
        print(f"  - images: {images.shape}")
        print(f"  - detections: 리스트 길이 {len(detections)}, 첫 번째 항목 길이 {len(detections[0])}")
        print(f"  - types: 리스트 길이 {len(types)}, 첫 번째 항목 길이 {len(types[0])}")
        print(f"  - names: {names}")
        
        # 클래스 분포 확인
        print("\n=== 클래스 분포 확인 ===")
        train_labels = []
        val_labels = []
        
        for data in train_dataset:
            train_labels.extend(data[2])  # types는 세 번째 요소
        
        for data in val_dataset:
            val_labels.extend(data[2])  # types는 세 번째 요소
        
        print("\nTrain 데이터셋 클래스 분포:")
        train_counter = Counter(train_labels)
        total_train = sum(train_counter.values())
        for k, v in sorted(train_counter.items()):
            print(f"클래스 {k}: {v}개 ({v/total_train*100:.1f}%)")
        
        print("\nVal 데이터셋 클래스 분포:")
        val_counter = Counter(val_labels)
        total_val = sum(val_counter.values())
        for k, v in sorted(val_counter.items()):
            print(f"클래스 {k}: {v}개 ({v/total_val*100:.1f}%)")
        
        print("\n✅ 데이터셋 로딩 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_dataset_loading()
    print(f"\n최종 결과: {'성공' if success else '실패'}") 