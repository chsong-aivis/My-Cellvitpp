#!/usr/bin/env python3
"""
DetectionDataset이 실제 학습 코드에서 요구하는 방식대로 잘 불러와지는지 확인하는 테스트 스크립트
"""
from cellvit.training.datasets.detection_dataset import DetectionDataset
from collections import Counter, defaultdict

def analyze_class_distribution(dataset):
    """데이터셋의 클래스 분포 분석"""
    all_types = []
    for i in range(len(dataset)):
        _, _, types, _ = dataset[i]
        all_types.extend(types)
    
    type_counter = Counter(all_types)
    total = sum(type_counter.values())
    
    print("\n클래스 분포:")
    for class_id, count in sorted(type_counter.items()):
        percentage = (count / total) * 100
        print(f"클래스 {class_id}: {count}개 ({percentage:.1f}%)")
    
    return type_counter

if __name__ == "__main__":
    dataset_path = './Dataset_Old'
    train_filelist = './Dataset_Old/splits/fold_0/train.csv'
    val_filelist = './Dataset_Old/splits/fold_0/val.csv'

    # 데이터셋 로드
    train_dataset = DetectionDataset(
        dataset_path=dataset_path,
        split="train",
        filelist_path=train_filelist
    )
    val_dataset = DetectionDataset(
        dataset_path=dataset_path,
        split="train",
        filelist_path=val_filelist
    )
    
    # 데이터셋 캐싱
    train_dataset.cache_dataset()
    val_dataset.cache_dataset()
    
    print(f"Train 샘플 수: {len(train_dataset)}")
    print(f"Val 샘플 수: {len(val_dataset)}")

    # 클래스 분포 분석
    print("\n=== Train 데이터셋 ===")
    train_dist = analyze_class_distribution(train_dataset)
    
    print("\n=== Validation 데이터셋 ===")
    val_dist = analyze_class_distribution(val_dataset)

    # 샘플 1개씩 확인
    for ds, name in zip([train_dataset, val_dataset], ["train", "val"]):
        if len(ds) > 0:
            img, detections, types, img_name = ds[0]
            print(f"\n[{name}] 첫 샘플: {img_name}")
            print(f"  - 이미지 shape: {img.shape}")
            print(f"  - 검출 개수: {len(detections)}")
            print(f"  - 타입 분포: {dict((t, types.count(t)) for t in set(types))}")
        else:
            print(f"[{name}] 데이터 없음!")

    # 원본 분포와 비교
    original_dist = {
        0: 61427,  # 12.4%
        1: 72227,  # 14.6%
        2: 33353,  # 6.7%
        3: 85415,  # 17.2%
        4: 243311  # 49.1%
    }
    
    print("\n=== 원본 vs 변환 후 분포 비교 ===")
    total_converted = sum((train_dist[i] + val_dist[i] for i in range(5)))
    total_original = sum(original_dist.values())
    
    print("\n클래스별 비교:")
    for class_id in range(5):
        orig_count = original_dist[class_id]
        orig_percent = (orig_count / total_original) * 100
        
        conv_count = train_dist[class_id] + val_dist[class_id]
        conv_percent = (conv_count / total_converted) * 100 if total_converted > 0 else 0
        
        print(f"클래스 {class_id}:")
        print(f"  - 원본: {orig_count}개 ({orig_percent:.1f}%)")
        print(f"  - 변환: {conv_count}개 ({conv_percent:.1f}%)")
        print(f"  - 차이: {conv_count - orig_count}개 ({conv_percent - orig_percent:.1f}%p)") 