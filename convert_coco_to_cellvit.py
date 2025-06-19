#!/usr/bin/env python3
"""
COCO 형식을 CellViT++ 형식으로 변환하는 스크립트
"""

import json
import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Tuple, Any
import random

class COCOToCellViTConverter:
    def __init__(
        self, 
        coco_json_path: str, 
        images_dir: str, 
        output_dir: str,
        val_ratio: float = 0.15,
        visualization_dir: str = None
    ):
        """
        COCO → CellViT++ 변환기 초기화
        
        Args:
            coco_json_path: COCO JSON 파일 경로
            images_dir: 이미지 폴더 경로
            output_dir: 출력 폴더 경로
            val_ratio: validation 데이터 비율 (0.0 ~ 1.0)
            visualization_dir: 시각화 결과 저장 폴더
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.visualization_dir = visualization_dir
        
        # 데이터 저장소
        self.coco_data = None
        self.images_info = {}
        self.annotations_info = {}
        self.categories_info = {}
        
        # 카테고리 ID 매핑 (9 → 4)
        self.category_mapping = {
            0: 0,  # 0
            1: 1,  # 1+
            2: 2,  # 2+
            3: 3,  # 3+
            9: 4   # nt → 4
        }
        
    def load_coco_data(self) -> bool:
        """COCO JSON 파일 로드"""
        try:
            with open(self.coco_json_path, 'r', encoding='utf-8') as f:
                self.coco_data = json.load(f)
            print(f"✅ COCO JSON 파일 로드 성공: {self.coco_json_path}")
            return True
        except Exception as e:
            print(f"❌ COCO JSON 파일 로드 실패: {e}")
            return False
    
    def prepare_directories(self):
        """출력 디렉토리 구조 생성"""
        # 기본 디렉토리
        dirs = [
            "train/images",
            "train/labels",
            "test/images",
            "test/labels",
            "splits/fold_0",
            "train_configs/ViT256"
        ]
        
        for d in dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)
        
        # 시각화 디렉토리
        if self.visualization_dir:
            os.makedirs(os.path.join(self.visualization_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.visualization_dir, "val"), exist_ok=True)
    
    def filter_and_process_data(self):
        """데이터 필터링 및 전처리"""
        if not self.coco_data:
            return None, None
        
        # 이미지별 어노테이션 수집
        image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            ann['category_id'] = self.category_mapping[ann['category_id']]  # 카테고리 ID 매핑
            image_annotations[img_id].append(ann)
        
        # 어노테이션이 있는 이미지만 선택
        valid_images = []
        for img in self.coco_data['images']:
            if img['id'] in image_annotations and len(image_annotations[img['id']]) > 0:
                valid_images.append((img, image_annotations[img['id']]))
        
        print(f"✅ 총 {len(valid_images)}개의 유효한 이미지 발견")
        return valid_images
    
    def convert_bbox_to_center(self, bbox: List[float]) -> Tuple[float, float]:
        """바운딩 박스를 중심점 좌표로 변환"""
        x, y, w, h = bbox
        return x + w/2, y + h/2
    
    def create_cellvit_label(self, annotations: List[Dict]) -> str:
        """CellViT++ 형식의 레이블 파일 생성"""
        lines = []
        for ann in annotations:
            x, y = self.convert_bbox_to_center(ann['bbox'])
            category_id = ann['category_id']
            lines.append(f"{int(x)},{int(y)},{category_id}")
        return "\n".join(lines)
    
    def create_label_map(self):
        """레이블 맵 파일 생성"""
        label_map = {
            0: "0",
            1: "1+",
            2: "2+",
            3: "3+",
            4: "nt"
        }
        
        with open(os.path.join(self.output_dir, "label_map.yaml"), 'w') as f:
            yaml.dump(label_map, f)
    
    def create_config_files(self, num_classes: int = 5):
        """훈련 설정 파일 생성"""
        config = {
            "logging": {
                "mode": "offline",
                "project": "cellvit++",
                "notes": "HER2 classification",
                "log_comment": "her2_classification",
                "wandb_dir": "./logs",
                "log_dir": "./logs",
                "level": "Debug"
            },
            "random_seed": 42,
            "gpu": 0,
            "data": {
                "dataset": "DetectionDataset",
                "dataset_path": self.output_dir,
                "normalize_stains_train": False,
                "normalize_stains_val": False,
                "num_classes": num_classes,
                "train_filelist": os.path.join(self.output_dir, "splits/fold_0/train.csv"),
                "val_filelist": os.path.join(self.output_dir, "splits/fold_0/val.csv"),
                "label_map": {i: name for i, name in enumerate(["0", "1+", "2+", "3+", "nt"])}
            },
            "model": {
                "hidden_dim": 256
            },
            "training": {
                "cache_cell_dataset": True,
                "batch_size": 64,
                "epochs": 50,
                "drop_rate": 0.1,
                "optimizer": "AdamW",
                "optimizer_hyperparameter": {
                    "lr": 0.0001,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999]
                },
                "early_stopping_patience": 10,
                "mixed_precision": True,
                "eval_every": 1,
                "scheduler": {
                    "scheduler_type": "exponential"
                }
            }
        }
        
        config_path = os.path.join(self.output_dir, "train_configs/ViT256/fold_0.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    def visualize_sample(self, image_path: str, annotations: List[Dict], output_path: str):
        """샘플 이미지와 어노테이션 시각화"""
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 색상 맵 정의
        colors = {
            0: (255, 0, 0),    # 빨강 (0)
            1: (0, 255, 0),    # 초록 (1+)
            2: (0, 0, 255),    # 파랑 (2+)
            3: (255, 255, 0),  # 노랑 (3+)
            4: (255, 0, 255)   # 마젠타 (nt)
        }
        
        # 어노테이션 그리기
        for ann in annotations:
            x, y = self.convert_bbox_to_center(ann['bbox'])
            category_id = ann['category_id']
            color = colors[category_id]
            
            # 중심점에 작은 원 그리기
            cv2.circle(image, (int(x), int(y)), 2, color, -1)
        
        # 이미지 저장
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def convert(self):
        """전체 변환 프로세스 실행"""
        print("🔄 COCO → CellViT++ 변환 시작")
        
        # 데이터 로드
        if not self.load_coco_data():
            return
        
        # 디렉토리 준비
        self.prepare_directories()
        
        # 데이터 필터링 및 전처리
        valid_images = self.filter_and_process_data()
        if not valid_images:
            return
        
        # Train/Val 분할
        train_images, val_images = train_test_split(
            valid_images,
            test_size=self.val_ratio,
            random_state=42
        )
        
        print(f"📊 데이터 분할:")
        print(f"   - 학습 데이터: {len(train_images)}개")
        print(f"   - 검증 데이터: {len(val_images)}개")
        
        # 데이터 변환 및 저장
        for phase, images in [("train", train_images), ("val", val_images)]:
            print(f"\n🔄 {phase} 데이터 변환 중...")
            
            # 파일 목록
            filenames = []
            
            for img_info, annotations in tqdm(images):
                # 이미지 복사
                src_path = os.path.join(self.images_dir, img_info['file_name'])
                dst_path = os.path.join(self.output_dir, "train/images", img_info['file_name'])
                shutil.copy2(src_path, dst_path)
                
                # 레이블 파일 생성
                label_content = self.create_cellvit_label(annotations)
                label_path = os.path.join(
                    self.output_dir,
                    "train/labels",
                    os.path.splitext(img_info['file_name'])[0] + ".csv"
                )
                with open(label_path, 'w') as f:
                    f.write(label_content)
                
                # 파일 목록에 추가
                filenames.append(os.path.splitext(img_info['file_name'])[0])
                
                # 시각화 (샘플)
                if self.visualization_dir and len(filenames) <= 5:
                    vis_path = os.path.join(
                        self.visualization_dir,
                        phase,
                        f"{os.path.splitext(img_info['file_name'])[0]}_vis.png"
                    )
                    self.visualize_sample(src_path, annotations, vis_path)
            
            # Split 파일 저장
            split_path = os.path.join(
                self.output_dir,
                "splits/fold_0",
                "train.csv" if phase == "train" else "val.csv"
            )
            with open(split_path, 'w') as f:
                f.write("\n".join(filenames))
        
        # 레이블 맵 생성
        self.create_label_map()
        
        # 설정 파일 생성
        self.create_config_files()
        
        print("\n✅ 변환 완료!")
        print(f"📁 결과물: {self.output_dir}")
        if self.visualization_dir:
            print(f"🎨 시각화 결과: {self.visualization_dir}")

def main():
    # 기본 경로
    coco_json_path = "/mnt/nas2/ihc_detection/data/her2/old_1217/coco.json"
    images_dir = "/mnt/nas2/DATA/HER2/old_1217/patches_sv/"
    output_dir = "./Dataset_Old"
    visualization_dir = "./Training_Example"
    
    # 변환기 초기화 및 실행
    converter = COCOToCellViTConverter(
        coco_json_path=coco_json_path,
        images_dir=images_dir,
        output_dir=output_dir,
        val_ratio=0.15,  # 15% for validation
        visualization_dir=visualization_dir
    )
    
    converter.convert()

if __name__ == "__main__":
    main() 