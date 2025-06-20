#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def analyze_validation_distribution():
    # Validation split 파일 읽기
    val_split = pd.read_csv("Dataset_Old/splits/fold_0/val.csv", header=None)
    val_files = val_split[0].tolist()
    print(f"Validation split에 총 {len(val_files)}개의 파일이 있습니다.")
    
    # 전체 클래스 분포를 저장할 Counter
    total_distribution = Counter()
    # 파일별 클래스 분포를 저장할 딕셔너리
    file_distribution = {}
    
    # 각 validation 파일의 레이블 분포 확인
    for val_file in tqdm(val_files, desc="Validation 파일 분석 중"):
        label_file = Path(f"Dataset_Old/train/labels/{val_file}.csv")
        try:
            # CSV 파일 읽기 (x, y, class 형식)
            df = pd.read_csv(label_file, header=None)
            # 클래스 컬럼(마지막 컬럼)의 분포 계산
            class_counts = Counter(df.iloc[:, 2])
            
            # 전체 분포에 추가
            total_distribution.update(class_counts)
            # 파일별 분포 저장
            file_distribution[label_file.stem] = dict(class_counts)
            
        except Exception as e:
            print(f"Error processing {label_file}: {str(e)}")
    
    # 결과 출력
    print("\n=== Validation 세트의 전체 클래스 분포 ===")
    total_samples = sum(total_distribution.values())
    for class_idx, count in sorted(total_distribution.items()):
        percentage = (count / total_samples) * 100
        print(f"클래스 {class_idx}: {count}개 ({percentage:.2f}%)")
    
    # 클래스별로 나타나는 파일 수 계산
    files_per_class = Counter()
    for file_dist in file_distribution.values():
        for class_idx in file_dist.keys():
            files_per_class[class_idx] += 1
    
    print("\n=== Validation 세트의 클래스별 파일 분포 ===")
    total_files = len(file_distribution)
    for class_idx, count in sorted(files_per_class.items()):
        percentage = (count / total_files) * 100
        print(f"클래스 {class_idx}를 포함하는 파일: {count}개 ({percentage:.2f}%)")
    
    # 클래스 조합 분석
    print("\n=== Validation 세트의 파일별 클래스 조합 ===")
    class_combinations = Counter(tuple(sorted(d.keys())) for d in file_distribution.values())
    for combination, count in class_combinations.most_common():
        percentage = (count / total_files) * 100
        print(f"클래스 조합 {combination}: {count}개 파일 ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_validation_distribution() 