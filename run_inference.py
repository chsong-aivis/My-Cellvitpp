#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, F1Score
import csv
import torchstain
from torch.utils.data import Dataset

# 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

from cellvit.training.evaluate.inference_cellvit_experiment_classifier import CellViTClassifierInferenceExperiment
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.utils.logger import Logger
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.config.config import BACKBONE_EMBED_DIM, CELL_IMAGE_SIZES
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.data.dataclass.wsi import WSIMetadata

class CustomDetectionDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, normalize_stains=False):
        """Custom dataset initialization that doesn't require the split directory structure"""
        super().__init__()  # Dataset의 init 호출
        self.transforms = transforms
        self.normalize_stains = normalize_stains
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.dataset_path = Path(dataset_path)
        self.image_path = self.dataset_path / "images"
        self.annotation_path = self.dataset_path / "labels"

        self.images = [
            f
            for f in sorted(self.image_path.glob("*"))
            if f.suffix in [".png", ".jpg", ".jpeg"]
        ]
        
        self.annotations = []
        for img_path in self.images:
            img_name = img_path.stem
            self.annotations.append(self.annotation_path / f"{img_name}.csv")

        self.cache_images = {}
        self.cache_annotations = {}
        
    def cache_dataset(self):
        """Cache the dataset in memory"""
        for img_path, annot_path in tqdm(
            zip(self.images, self.annotations), total=len(self.images), desc="Caching dataset"
        ):
            img = Image.open(img_path)
            img = img.convert("RGB")
            self.cache_images[img_path.stem] = img

            with open(annot_path, "r") as file:
                reader = csv.reader(file)
                cell_annot = list(reader)
                # Convert strings to integers
                cell_annot = [[int(float(x)), int(float(y)), int(float(t))] for x, y, t in cell_annot]
            self.cache_annotations[img_path.stem] = cell_annot

    def __getitem__(self, index: int):
        """Get item from dataset with modified keypoint format

        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, list, list, str]:
            * Image
            * List of detections
            * List of types
            * Name of the Patch
        """
        img_path = self.images[index]
        img_name = img_path.stem
        img = self.cache_images[img_name]
        cell_annot = self.cache_annotations[img_name]
        
        # 각 keypoint에 angle=0, scale=1 추가
        detections = [(int(x), int(y), 0, 1) for x, y, _ in cell_annot]
        types = [int(t) for _, _, t in cell_annot]

        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))

        img = np.array(img).astype(np.uint8)

        if self.transforms:
            transformed = self.transforms(image=img, keypoints=detections)
            img = transformed["image"]
            detections = transformed["keypoints"]
            # keypoint에서 다시 (x, y)만 추출
            detections = [(int(x), int(y)) for x, y, _, _ in detections]
            types = [types[idx] for idx, _ in enumerate(detections)]

        return img, detections, types, img_name

class CustomInferenceExperiment(CellViTClassifierInferenceExperiment):
    def _load_dataset(self, transforms, normalize_stains):
        """데이터셋 로드 구현"""
        # 이미지 크기를 512x512로 맞추는 transform 추가
        if transforms is None:
            transforms = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy'))
        else:
            # 기존 transform에 Resize 추가
            transforms.transforms.insert(0, A.Resize(512, 512))
            
        dataset = CustomDetectionDataset(
            dataset_path=self.dataset_path,
            transforms=transforms,
            normalize_stains=normalize_stains
        )
        # 데이터셋 캐싱
        dataset.cache_dataset()
        return dataset
    
    def run_inference(self):
        """추론 실행 구현"""
        # 결과 저장 디렉토리 생성
        results_dir = Path("inference_results")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # 데이터로더 설정
        dataloader = torch.utils.data.DataLoader(
            self.inference_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=DetectionDataset.collate_batch
        )
        
        # 모델을 평가 모드로 설정
        self.cellvit_model.eval()
        self.model.eval()
        
        # 메트릭 초기화 (학습 코드와 동일하게)
        metrics = {
            'auroc': AUROC(task="multiclass", num_classes=5).to(self.device),
            'accuracy': Accuracy(task="multiclass", num_classes=5).to(self.device),
            'f1': F1Score(task="multiclass", num_classes=5).to(self.device),
            'ap': AveragePrecision(task="multiclass", num_classes=5).to(self.device)
        }
        
        all_preds = []
        all_labels = []
        all_image_names = []
        
        # 후처리기 초기화
        wsi_metadata = WSIMetadata(
            name="validation_dataset",
            slide_path=str(self.dataset_path),
            metadata={
                "mpp": 0.25,  # 0.25 microns per pixel (40x)
                "objective_power": 40,
                "vendor": "custom",
                "mpp_x": 0.25,
                "mpp_y": 0.25,
                "image_size": (512, 512),
                "tile_size": (512, 512),
            }
        )
        postprocessor = DetectionCellPostProcessorCupy(
            wsi=wsi_metadata,
            nr_types=6,  # PanNuke 데이터셋의 클래스 수
            resolution=0.25,  # 40x = 0.25 microns per pixel
            binary=False,
            gt=False
        )
        
        # 추론 실행
        self.logger.info("Running inference on validation dataset...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                images, detections, types, names = batch
                images = images.to(self.device)
                
                # Stage 1: Cell Detection & Feature Extraction
                cell_pred_dict, cell_gt_dict, metrics_dict, _, _, _ = self._get_cellvit_result(
                    images=images,
                    cell_gt_batch=detections,
                    types_batch=types,
                    image_names=names,
                    postprocessor=postprocessor
                )
                
                # Stage 2: Classification
                classifier_results = self._get_classifier_result(
                    extracted_cells=cell_gt_dict,  # cell_pred_dict 대신 cell_gt_dict 사용
                    threshold=0.5
                )
                
                if classifier_results["predictions"].size(0) > 0:
                    # 결과 저장 (numeric 형태로)
                    all_preds.append(classifier_results["probabilities"])
                    all_labels.append(classifier_results["gt"])
                    all_image_names.extend([names[0]] * len(classifier_results["predictions"]))
        
        # 전체 결과에 대해 메트릭 계산
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            results = {
                'auroc': metrics['auroc'](all_preds, all_labels),
                'accuracy': metrics['accuracy'](all_preds, all_labels),
                'f1': metrics['f1'](all_preds, all_labels),
                'ap': metrics['ap'](all_preds, all_labels)
            }
            
            # 결과를 string label로 변환하여 저장
            pred_classes = torch.argmax(all_preds, dim=1).cpu().numpy()
            true_classes = all_labels.cpu().numpy()
            
            label_map = {
                0: '0',
                1: '1+',
                2: '2+',
                3: '3+',
                4: 'nt'
            }
            
            results_df = pd.DataFrame({
                'Image': all_image_names,
                'Prediction': [label_map[p] for p in pred_classes],
                'Ground Truth': [label_map[t] for t in true_classes]
            })
            
            # 결과 저장
            results_df.to_csv(results_dir / "inference_results.csv", index=False)
            
            # 분류 리포트 생성
            class_names = list(label_map.values())
            report = classification_report(
                results_df['Ground Truth'], 
                results_df['Prediction'],
                target_names=class_names,
                labels=class_names,
                output_dict=True
            )
            
            # 혼동 행렬 생성 및 저장
            cm = confusion_matrix(
                results_df['Ground Truth'], 
                results_df['Prediction'],
                labels=class_names
            )
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(results_dir / "confusion_matrix.png")
            plt.close()
            
            # 결과 요약
            self.logger.info("\nClassification Report:")
            self.logger.info("\n" + classification_report(
                results_df['Ground Truth'],
                results_df['Prediction'],
                target_names=class_names,
                labels=class_names
            ))
            
            # 메트릭 결과 출력
            self.logger.info("\nMetrics:")
            for metric_name, metric_value in results.items():
                self.logger.info(f"{metric_name}: {metric_value:.4f}")
            
            self.logger.info(f"\nResults saved to {results_dir}")
            
            return results_df, results, cm
        else:
            self.logger.error("No predictions were made!")
            return None, None, None

def main():
    """메인 실행 함수"""
    # 모델 체크포인트 경로
    logdir = "logs_sam_bs128_noweights/2025-06-19T131216_custom training with batch size 128 (no class weights)"
    cellvit_path = "checkpoints/classifier/CellViT-SAM-H-x40-AMP-002.pth"
    dataset_path = "Validation_Dataset"
    
    try:
        # 추론 실험 설정 및 실행
        experiment = CustomInferenceExperiment(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=False,
            gpu=0
        )
        results_df, metrics, cm = experiment.run_inference()
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 