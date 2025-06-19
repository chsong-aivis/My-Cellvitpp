#!/usr/bin/env python3
"""
COCO 형태의 데이터셋 분석 스크립트
CellViT++ 학습을 위한 데이터셋 구조 파악용
"""

import json
import os
import cv2
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm

class COCODatasetAnalyzer:
    def __init__(self, coco_json_path: str, images_dir: str):
        """
        COCO 데이터셋 분석기 초기화
        
        Args:
            coco_json_path: COCO JSON 파일 경로
            images_dir: 이미지 폴더 경로
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.coco_data = None
        self.images_info = {}
        self.annotations_info = {}
        self.categories_info = {}
        
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
    
    def analyze_basic_info(self):
        """기본 정보 분석"""
        if not self.coco_data:
            print("❌ COCO 데이터가 로드되지 않았습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 COCO 데이터셋 기본 정보")
        print("="*60)
        
        # 기본 구조 정보
        print(f"📁 JSON 키: {list(self.coco_data.keys())}")
        
        # 이미지 정보
        if 'images' in self.coco_data:
            images = self.coco_data['images']
            print(f"🖼️  총 이미지 수: {len(images)}")
            
            # 이미지 크기 통계
            widths = [img['width'] for img in images]
            heights = [img['height'] for img in images]
            print(f"📏 이미지 크기 통계:")
            print(f"   - 너비: {min(widths)} ~ {max(widths)} (평균: {np.mean(widths):.1f})")
            print(f"   - 높이: {min(heights)} ~ {max(heights)} (평균: {np.mean(heights):.1f})")
            
            # 파일 형식 분석
            file_extensions = [Path(img['file_name']).suffix.lower() for img in images]
            ext_counter = Counter(file_extensions)
            print(f"📄 파일 형식: {dict(ext_counter)}")
            
            # 이미지 정보 저장
            self.images_info = {img['id']: img for img in images}
        
        # 카테고리 정보
        if 'categories' in self.coco_data:
            categories = self.coco_data['categories']
            print(f"🏷️  총 카테고리 수: {len(categories)}")
            print("📋 카테고리 목록:")
            for cat in categories:
                print(f"   - ID {cat['id']}: {cat['name']}")
            
            self.categories_info = {cat['id']: cat for cat in categories}
        
        # 어노테이션 정보
        if 'annotations' in self.coco_data:
            annotations = self.coco_data['annotations']
            print(f"🎯 총 어노테이션 수: {len(annotations)}")
            
            # 어노테이션 타입 분석
            annotation_types = [ann.get('segmentation', 'bbox') for ann in annotations]
            type_counter = Counter(annotation_types)
            print(f"📝 어노테이션 타입: {dict(type_counter)}")
            
            self.annotations_info = annotations
    
    def analyze_annotations(self):
        """어노테이션 상세 분석"""
        if not self.annotations_info:
            print("❌ 어노테이션 정보가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("🎯 어노테이션 상세 분석")
        print("="*60)
        
        annotations = self.annotations_info
        
        # 카테고리별 어노테이션 수
        category_counts = Counter([ann['category_id'] for ann in annotations])
        print("📊 카테고리별 어노테이션 수:")
        for cat_id, count in category_counts.most_common():
            cat_name = self.categories_info.get(cat_id, {}).get('name', f'Unknown_{cat_id}')
            print(f"   - {cat_name} (ID: {cat_id}): {count}개")
        
        # 이미지별 어노테이션 수
        image_annotation_counts = Counter([ann['image_id'] for ann in annotations])
        print(f"\n📈 이미지별 어노테이션 수 통계:")
        print(f"   - 최소: {min(image_annotation_counts.values())}")
        print(f"   - 최대: {max(image_annotation_counts.values())}")
        print(f"   - 평균: {np.mean(list(image_annotation_counts.values())):.1f}")
        print(f"   - 중앙값: {np.median(list(image_annotation_counts.values())):.1f}")
        
        # 바운딩 박스 크기 분석 (bbox가 있는 경우)
        bbox_annotations = [ann for ann in annotations if 'bbox' in ann]
        if bbox_annotations:
            widths = [bbox[2] for bbox in [ann['bbox'] for ann in bbox_annotations]]
            heights = [bbox[3] for bbox in [ann['bbox'] for ann in bbox_annotations]]
            areas = [bbox[2] * bbox[3] for bbox in [ann['bbox'] for ann in bbox_annotations]]
            
            print(f"\n📏 바운딩 박스 크기 통계:")
            print(f"   - 너비: {min(widths):.1f} ~ {max(widths):.1f} (평균: {np.mean(widths):.1f})")
            print(f"   - 높이: {min(heights):.1f} ~ {max(heights):.1f} (평균: {np.mean(heights):.1f})")
            print(f"   - 면적: {min(areas):.1f} ~ {max(areas):.1f} (평균: {np.mean(areas):.1f})")
    
    def check_image_files(self):
        """이미지 파일 존재 여부 확인"""
        if not self.images_info:
            print("❌ 이미지 정보가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("🔍 이미지 파일 존재 여부 확인")
        print("="*60)
        
        missing_files = []
        existing_files = []
        
        for img_id, img_info in self.images_info.items():
            file_path = os.path.join(self.images_dir, img_info['file_name'])
            if os.path.exists(file_path):
                existing_files.append(file_path)
            else:
                missing_files.append(img_info['file_name'])
        
        print(f"✅ 존재하는 파일: {len(existing_files)}개")
        print(f"❌ 누락된 파일: {len(missing_files)}개")
        
        if missing_files:
            print(f"\n📋 누락된 파일 목록 (처음 10개):")
            for file in missing_files[:10]:
                print(f"   - {file}")
            if len(missing_files) > 10:
                print(f"   ... 외 {len(missing_files) - 10}개")
        
        return existing_files, missing_files
    
    def analyze_image_samples(self, num_samples: int = 5):
        """샘플 이미지 분석"""
        if not self.images_info:
            print("❌ 이미지 정보가 없습니다.")
            return
        
        print(f"\n" + "="*60)
        print(f"🔬 샘플 이미지 분석 ({num_samples}개)")
        print("="*60)
        
        # 랜덤 샘플 선택
        sample_images = list(self.images_info.values())[:num_samples]
        
        for i, img_info in enumerate(sample_images, 1):
            print(f"\n📸 샘플 {i}: {img_info['file_name']}")
            print(f"   - 크기: {img_info['width']} x {img_info['height']}")
            print(f"   - ID: {img_info['id']}")
            
            # 해당 이미지의 어노테이션 찾기
            img_annotations = [ann for ann in self.annotations_info if ann['image_id'] == img_info['id']]
            print(f"   - 어노테이션 수: {len(img_annotations)}")
            
            if img_annotations:
                # 카테고리별 어노테이션 수
                cat_counts = Counter([ann['category_id'] for ann in img_annotations])
                print(f"   - 카테고리별 분포:")
                for cat_id, count in cat_counts.items():
                    cat_name = self.categories_info.get(cat_id, {}).get('name', f'Unknown_{cat_id}')
                    print(f"     * {cat_name}: {count}개")
    
    def convert_to_cellvit_format(self, output_dir: str = "./cellvit_dataset"):
        """CellViT++ 형식으로 변환 계획"""
        print("\n" + "="*60)
        print("🔄 CellViT++ 형식 변환 계획")
        print("="*60)
        
        if not self.coco_data:
            print("❌ COCO 데이터가 로드되지 않았습니다.")
            return
        
        print(f"📁 출력 디렉토리: {output_dir}")
        print("\n📋 변환 계획:")
        print("1. 폴더 구조 생성")
        print("   - train/images/")
        print("   - train/labels/")
        print("   - test/images/")
        print("   - test/labels/")
        print("   - splits/fold_0/")
        print("   - train_configs/ViT256/")
        
        print("\n2. 데이터 변환:")
        print("   - COCO bbox → CellViT++ 중심점 좌표")
        print("   - 카테고리 ID → 0부터 시작하는 연속 정수")
        print("   - 이미지 크기 확인 및 조정")
        
        print("\n3. 필요한 작업:")
        print("   - 이미지 크기가 32의 배수인지 확인")
        print("   - 레이블 맵 생성")
        print("   - 데이터 분할 (train/val/test)")
        print("   - 훈련 설정 파일 생성")
        
        # 레이블 맵 생성 예시
        if self.categories_info:
            print(f"\n📝 레이블 맵 예시:")
            for i, (cat_id, cat_info) in enumerate(self.categories_info.items()):
                print(f"   {i}: \"{cat_info['name']}\"")
    
    def generate_summary_report(self):
        """요약 리포트 생성"""
        print("\n" + "="*60)
        print("📊 데이터셋 요약 리포트")
        print("="*60)
        
        if not self.coco_data:
            print("❌ COCO 데이터가 로드되지 않았습니다.")
            return
        
        summary = {
            "총 이미지 수": len(self.coco_data.get('images', [])),
            "총 어노테이션 수": len(self.coco_data.get('annotations', [])),
            "총 카테고리 수": len(self.coco_data.get('categories', [])),
            "이미지 크기 범위": f"{min([img['width'] for img in self.coco_data.get('images', [])])} ~ {max([img['width'] for img in self.coco_data.get('images', [])])}",
            "카테고리 목록": [cat['name'] for cat in self.coco_data.get('categories', [])]
        }
        
        for key, value in summary.items():
            print(f"📋 {key}: {value}")
        
        print(f"\n🎯 CellViT++ 변환 준비도:")
        
        # 이미지 크기 확인
        images = self.coco_data.get('images', [])
        valid_sizes = [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]
        
        width_valid = all(img['width'] in valid_sizes for img in images)
        height_valid = all(img['height'] in valid_sizes for img in images)
        
        if width_valid and height_valid:
            print("   ✅ 이미지 크기: CellViT++ 호환")
        else:
            print("   ⚠️  이미지 크기: 리사이즈 필요")
        
        if len(self.coco_data.get('categories', [])) > 0:
            print("   ✅ 카테고리 정보: 존재")
        else:
            print("   ❌ 카테고리 정보: 없음")
        
        if len(self.coco_data.get('annotations', [])) > 0:
            print("   ✅ 어노테이션: 존재")
        else:
            print("   ❌ 어노테이션: 없음")

def main():
    parser = argparse.ArgumentParser(description='COCO 데이터셋 분석')
    parser.add_argument('--coco_json', type=str, required=True, help='COCO JSON 파일 경로')
    parser.add_argument('--images_dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--samples', type=int, default=5, help='분석할 샘플 이미지 수')
    
    args = parser.parse_args()
    
    # 분석기 초기화
    analyzer = COCODatasetAnalyzer(args.coco_json, args.images_dir)
    
    # 데이터 로드
    if not analyzer.load_coco_data():
        return
    
    # 분석 실행
    analyzer.analyze_basic_info()
    analyzer.analyze_annotations()
    analyzer.check_image_files()
    analyzer.analyze_image_samples(args.samples)
    analyzer.convert_to_cellvit_format()
    analyzer.generate_summary_report()

if __name__ == "__main__":
    # 직접 실행 시 기본 경로 사용
    coco_json_path = "/mnt/nas2/ihc_detection/data/her2/old_1217/coco.json"
    images_dir = "/mnt/nas2/DATA/HER2/old_1217/patches_sv/"
    
    print("🔍 COCO 데이터셋 분석 시작")
    print(f"📄 COCO JSON: {coco_json_path}")
    print(f"🖼️  이미지 폴더: {images_dir}")
    
    analyzer = COCODatasetAnalyzer(coco_json_path, images_dir)
    
    # 데이터 로드
    if not analyzer.load_coco_data():
        print("❌ 데이터 로드 실패")
        exit(1)
    
    # 분석 실행
    analyzer.analyze_basic_info()
    analyzer.analyze_annotations()
    analyzer.check_image_files()
    analyzer.analyze_image_samples(5)
    analyzer.convert_to_cellvit_format()
    analyzer.generate_summary_report()
    
    print("\n✅ 분석 완료!") 