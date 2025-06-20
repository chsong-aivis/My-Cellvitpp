# -*- coding: utf-8 -*-
# Custom Inference Code for Test Data
#
# Based on CellViT's inference code

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)

import argparse
import json
import yaml
from pathlib import Path
from typing import Callable, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)

from cellvit.config.config import CELL_IMAGE_SIZES
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.classifier.linear_classifier import LinearClassifier

class CustomDataset(Dataset):
    """Custom dataset for our data"""
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str = "val",
        normalize_stains: bool = False,
        transforms: Callable = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.normalize_stains = normalize_stains
        self.transforms = transforms
        
        # Load split file
        split_file = self.dataset_path / "splits" / "fold_0" / f"{split}.csv"
        self.image_list = pd.read_csv(split_file, header=None)[0].tolist()
        
        # Load label map
        with open(self.dataset_path / "label_map.yaml", "r") as f:
            self.label_map = yaml.safe_load(f)
        
        self.images_dir = self.dataset_path / "train" / "images"
        self.labels_dir = self.dataset_path / "train" / "labels"
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        
        # Load image
        image_path = self.images_dir / f"{image_name}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label_path = self.labels_dir / f"{image_name}.csv"
        label_df = pd.read_csv(label_path, header=None, names=['x', 'y', 'class'])
        # Get the most common class (majority voting)
        label = label_df['class'].mode().iloc[0]
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        
        return {
            "image": image,
            "label": label,
            "image_name": image_name
        }

class CellViTInfExpCustom(CellViTClassifierInferenceExperiment):
    """Custom inference experiment for our data"""
    def __init__(
        self,
        logdir: Union[Path, str],
        dataset_path: Union[Path, str],
        cellvit_path: Union[Path, str],
        input_shape: Tuple[int, int],
        device: str = "cuda",
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Get GPU number from device string
        if ":" in device:
            gpu = int(device.split(":")[-1])
        else:
            gpu = 0
        
        super().__init__(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=False,
            gpu=gpu,
        )
        
        # Load backbone model
        self.backbone = CellViTSAM(
            model_path=cellvit_path,
            num_nuclei_classes=2,  # Binary segmentation
            num_tissue_classes=5,  # Our 5 classes
            vit_structure="SAM-H",
            drop_rate=0.0,
            regression_loss=False,
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Override classifier head to match hidden_dim
        self.model = LinearClassifier(
            embed_dim=1280,  # Changed to match SAM-H's embed_dim
            num_classes=5,
            drop_rate=0.0,
        )
        self.model.load_state_dict(torch.load(self.model_path)["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
    def _load_dataset(
        self,
        transforms: Callable,
        normalize_stains: bool,
    ) -> Dataset:
        """Load custom dataset"""
        dataset = CustomDataset(
            dataset_path=self.dataset_path,
            split="val",
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        return dataset
    
    def run_inference(self) -> None:
        """Run inference on the dataset"""
        self.logger.info("Starting inference...")
        
        # Initialize metrics
        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=5).to(self.device),
            "f1": F1Score(task="multiclass", num_classes=5).to(self.device),
            "precision": Precision(task="multiclass", num_classes=5).to(self.device),
            "recall": Recall(task="multiclass", num_classes=5).to(self.device),
            "auroc": AUROC(task="multiclass", num_classes=5).to(self.device),
            "ap": AveragePrecision(task="multiclass", num_classes=5).to(self.device),
        }
        
        # Create dataloader
        dataloader = DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # Run inference
        all_preds = []
        all_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Running inference"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                image_names = batch["image_name"]
                
                # Extract features using backbone
                outputs = self.backbone(images, retrieve_tokens=True)
                features = outputs["tokens"]
                print("Features shape:", features.shape)
                
                # Average over spatial dimensions
                features = features.mean(dim=(2, 3))  # Average over spatial dimensions
                print("Features after mean:", features.shape)
                
                # Get predictions
                outputs = self.model(features)
                preds = torch.softmax(outputs, dim=1)
                
                # Update metrics
                for metric in metrics.values():
                    metric.update(preds, labels)
                
                # Store predictions and labels
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        # Compute final metrics
        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute().item()
        
        # Save results
        results_path = Path(self.logdir) / "test_results"
        results_path.mkdir(exist_ok=True)
        
        with open(results_path / "metrics.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Save predictions
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        predictions_df = pd.DataFrame({
            "image_name": all_image_names,
            "true_label": all_labels,
            "pred_0": all_preds[:, 0],
            "pred_1": all_preds[:, 1],
            "pred_2": all_preds[:, 2],
            "pred_3": all_preds[:, 3],
            "pred_4": all_preds[:, 4],
        })
        predictions_df.to_csv(results_path / "predictions.csv", index=False)
        
        self.logger.info(f"Results saved to {results_path}")
        self.logger.info("Metrics:")
        for name, value in results.items():
            self.logger.info(f"{name}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cellvit_path", type=str, required=True)
    parser.add_argument("--input_shape", type=int, nargs=2, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    experiment = CellViTInfExpCustom(
        logdir=args.logdir,
        dataset_path=args.dataset_path,
        cellvit_path=args.cellvit_path,
        input_shape=args.input_shape,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    experiment.run_inference()

if __name__ == "__main__":
    main() 