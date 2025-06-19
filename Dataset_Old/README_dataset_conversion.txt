# CellViT++ 데이터셋 변환 및 분석 코드 설명

## 1. analyze_coco_dataset.py
- **목적**: COCO 형식의 데이터셋을 분석하여 전체 구조, 카테고리, 어노테이션 분포, 이미지 크기, 파일 존재 여부 등을 파악합니다.
- **주요 기능**:
  - COCO JSON 파일 구조 및 통계 출력
  - 카테고리별/이미지별 어노테이션 분포 분석
  - 바운딩 박스 크기 통계
  - 이미지 파일 누락 여부 확인
  - 샘플 이미지별 어노테이션 분포 출력
  - CellViT++ 변환 가능성 및 준비도 요약
- **사용법**:
  ```bash
  python analyze_coco_dataset.py
  ```
  (경로는 코드 내에 기본값으로 지정되어 있음)

## 2. convert_coco_to_cellvit.py
- **목적**: COCO 형식의 데이터셋을 CellViT++ 학습에 맞는 폴더/파일 구조로 변환합니다.
- **주요 기능**:
  - 어노테이션이 없는 이미지는 제외
  - category_id가 9인 경우 4로 변환
  - 바운딩 박스 중심점 좌표로 변환하여 레이블 생성
  - 학습/검증 데이터 분할 (기본 85:15)
  - CellViT++ 폴더 구조로 이미지/레이블/분할 파일/설정 파일/레이블맵 생성
  - 학습/검증 데이터 샘플 시각화 (Training_Example 폴더)
- **사용법**:
  ```bash
  python convert_coco_to_cellvit.py
  ```
  (경로는 코드 내에 기본값으로 지정되어 있음)

## 3. 결과 폴더 구조
- `Dataset_Old/` : 변환된 CellViT++ 데이터셋
  - `train/images/`, `train/labels/` : 학습 이미지/레이블
  - `splits/fold_0/train.csv`, `val.csv` : 분할 파일
  - `label_map.yaml` : 레이블 맵
  - `train_configs/ViT256/fold_0.yaml` : 학습 설정 파일
- `Training_Example/` : 학습/검증 데이터 샘플 시각화 이미지

---

> 이 파일은 데이터셋 변환 및 분석 코드의 목적, 기능, 사용법, 결과 구조를 요약한 안내문입니다. 나중에 데이터셋 준비/학습 시 참고하세요. 