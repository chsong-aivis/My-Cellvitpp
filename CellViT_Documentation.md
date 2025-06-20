# CellViT++ Project Documentation

## Overview
CellViT++는 세포 검출 및 분류를 위한 Vision Transformer 기반 프레임워크입니다. 본 문서는 프로젝트의 구조와 사용법을 설명합니다.

---

## Model Architecture

### Backbone Models
CellViT++는 다양한 버전의 Vision Transformer 기반 백본을 제공합니다:
- CellViT256
- CellViTSAM
- CellViTUNI
- CellViTVirchow
- CellViTVirchow2

### Classifier Head
- 구조: Linear Classifier
- 입력: Backbone의 hidden dimension
- 출력: 데이터셋별 클래스 수

---

## Pre-trained Models

### Backbone Checkpoints
```
📁 checkpoints/
└── cellvit_sam_h.pth  # Google Drive에서 다운로드 필요
```

### Classifier Checkpoints
```
📁 checkpoints/classifier/sam-h/
├── consep.pth         # 4 classes: Other, Inflammatory, Epithelial, Spindle-Shaped
├── lizard.pth        # 6 classes: Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective
├── midog.pth         # 2 classes: Mitotic, Non-Mitotic
├── nucls_main.pth
├── nucls_super.pth
├── ocelot.pth        # 2 classes: Other Cell, Tumor Cell
└── panoptils.pth     # 4 classes: Other, Epithelial, Stromal, TILs
```

---

## Supported Datasets

### Built-in Datasets
- CoNSeP
- Ocelot
- SegPath
- MIDOG
- NuCLS
- Panoptils
- Lizard

### Dataset Features
- Albumentations 기반 데이터 증강
- 정규화 옵션
- 데이터셋별 전용 Dataset 클래스

---

## Training Process

### Basic Training Flow
```python
# 1. Load backbone model
cellvit_model = CellViT256.from_pretrained(checkpoint_path)
cellvit_model.eval()  # Feature extractor mode

# 2. Initialize classifier
classifier = LinearClassifier(
    input_dim=cellvit_model.hidden_dim,
    num_classes=num_classes
)

# 3. Setup training
optimizer = AdamW(classifier.parameters())
criterion = CrossEntropyLoss(weight=class_weights)

# 4. Initialize trainer and train
trainer = CellViTHeadTrainer(
    model=classifier,
    cellvit_model=cellvit_model,
    loss_fn=criterion,
    optimizer=optimizer
)
```

### Key Training Features
- ✅ Mixed Precision Training
- ✅ Gradient Accumulation
- ✅ Weighted Loss (클래스 불균형 처리)
- ✅ Early Stopping
- ✅ Learning Rate Scheduling
- ✅ WandB 모니터링

---

## Configuration

### Example Configuration File
```yaml
# Model configuration
model:
  name: "cellvit"
  backbone: "vit_h"
  pretrained_path: "checkpoints/cellvit_sam_h.pth"
  num_classes: 5

# Training configuration
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  optimizer: "AdamW"
  scheduler: "cosine"
  class_weights: [...]

# Data configuration
data:
  dataset_path: "Dataset_Old"
  train_filelist: "splits/train_split.csv"
  val_filelist: "splits/val_split.csv"
  normalize_stains: true
  augmentation:
    enabled: true
```

---

## HER2 Dataset Application

### Dataset Characteristics
| Class | Distribution |
|-------|-------------|
| 0     | 12.4%       |
| 1+    | 14.6%       |
| 2+    | 6.7%        |
| 3+    | 17.2%       |
| NT    | 49.1%       |

### Recommended Settings
1. **Base Model**: consep 분류기 기반 fine-tuning
2. **Loss Function**: 클래스 불균형을 고려한 weighted loss
3. **Data Augmentation**: 일반화 성능 향상을 위한 증강 기법 적용

---

## Project Structure
```
CellViT-plus-plus/
├── cellvit/
│   ├── models/          # 모델 구현
│   ├── training/        # 학습 관련 코드
│   └── utils/           # 유틸리티 함수
├── checkpoints/         # 모델 체크포인트
│   └── classifier/
│       └── sam-h/
├── configs/             # 설정 파일
├── logs/                # 학습 로그
└── Dataset_Old/         # 데이터셋
```

---

## Requirements
- CUDA-capable GPU (24GB VRAM 이상 권장)
- 32GB RAM 이상
- 30GB 디스크 공간
- 16 CPU 코어 이상

---

## References
- [CellViT++ GitHub Repository](https://github.com/TIO-IKIM/CellViT-plus-plus)
- [Original Paper](https://arxiv.org/abs/2501.05269) 