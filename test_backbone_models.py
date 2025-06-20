#!/usr/bin/env python3
"""
Backbone 모델 테스트 스크립트
다양한 CellViT Backbone 모델들을 로드하고 예시 데이터로 테스트하여
모델의 역할, 입력/출력 형태, 성능을 분석합니다.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import yaml
from tqdm import tqdm
import pandas as pd

# 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.data.dataclass.wsi import WSIMetadata
from cellvit.utils.logger import Logger

class BackboneTester:
    def __init__(self, device='cuda'):
        self.device = device
        self.logger = Logger(level="INFO").create_logger()
        self.results = {}
        
    def load_backbone_model(self, model_type, checkpoint_path):
        """Backbone 모델 로드"""
        self.logger.info(f"Loading {model_type} model from {checkpoint_path}")
        
        if model_type == "CellViT-SAM-H":
            model = CellViTSAM(
                model_path=checkpoint_path,
                num_nuclei_classes=6,  # PanNuke 클래스 수
                num_tissue_classes=0,  # 조직 분류 비활성화
                vit_structure="SAM-H",
                drop_rate=0.0
            )
        elif model_type == "CellViT-256":
            model = CellViT256(
                num_nuclei_classes=6,
                num_tissue_classes=0,
                drop_rate=0.0
            )
        elif model_type == "CellViT-Virchow":
            model = CellViTVirchow(
                num_nuclei_classes=6,
                num_tissue_classes=0,
                drop_rate=0.0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 체크포인트 로드
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def load_test_data(self, data_path):
        """테스트 데이터 로드"""
        self.logger.info(f"Loading test data from {data_path}")
        
        images = []
        labels = []
        
        image_dir = Path(data_path) / "images"
        label_dir = Path(data_path) / "labels"
        
        for img_file in sorted(image_dir.glob("*.png")):
            # 이미지 로드
            img = Image.open(img_file).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
            
            # 라벨 로드
            label_file = label_dir / f"{img_file.stem}.csv"
            if label_file.exists():
                df = pd.read_csv(label_file, header=None)
                labels.append(df.values)
            else:
                labels.append(np.array([]))
        
        return images, labels
    
    def preprocess_image(self, image, target_size=1024):
        """이미지 전처리"""
        # 이미지 리사이즈
        img = Image.fromarray(image)
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # 정규화
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW 형태
        
        return img_tensor.to(self.device)
    
    def run_inference(self, model, image_tensor):
        """추론 실행"""
        with torch.no_grad():
            start_time = time.time()
            outputs = model(image_tensor, retrieve_tokens=True)
            inference_time = time.time() - start_time
        
        return outputs, inference_time
    
    def analyze_outputs(self, outputs, model_name):
        """모델 출력 분석"""
        analysis = {
            'model_name': model_name,
            'output_keys': list(outputs.keys()),
            'output_shapes': {},
            'output_stats': {}
        }
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                analysis['output_shapes'][key] = list(value.shape)
                analysis['output_stats'][key] = {
                    'min': float(value.min()),
                    'max': float(value.max()),
                    'mean': float(value.mean()),
                    'std': float(value.std())
                }
        
        return analysis
    
    def visualize_results(self, image, outputs, save_path):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Backbone Model Outputs', fontsize=16)
        
        # 원본 이미지
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 세포 검출 결과
        if 'nuclei_binary_map' in outputs:
            binary_map = outputs['nuclei_binary_map'][0, 1].cpu().numpy()  # 세포 확률 맵
            axes[0, 1].imshow(binary_map, cmap='hot')
            axes[0, 1].set_title('Cell Detection Map')
            axes[0, 1].axis('off')
        
        # 세포 타입 맵
        if 'nuclei_type_map' in outputs:
            type_map = outputs['nuclei_type_map'][0].cpu().numpy()
            type_map_max = np.argmax(type_map, axis=0)
            axes[0, 2].imshow(type_map_max, cmap='tab10')
            axes[0, 2].set_title('Cell Type Map')
            axes[0, 2].axis('off')
        
        # HV 맵
        if 'hv_map' in outputs:
            hv_map = outputs['hv_map'][0].cpu().numpy()
            axes[1, 0].imshow(hv_map[0], cmap='RdBu')
            axes[1, 0].set_title('H Map')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(hv_map[1], cmap='RdBu')
            axes[1, 1].set_title('V Map')
            axes[1, 1].axis('off')
        
        # 토큰 시각화 (첫 번째 채널)
        if 'tokens' in outputs:
            tokens = outputs['tokens'][0, :, 0].cpu().numpy()
            token_map = tokens.reshape(int(np.sqrt(tokens.shape[0])), -1)
            axes[1, 2].imshow(token_map, cmap='viridis')
            axes[1, 2].set_title('Token Map (First Channel)')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_backbone_models(self, test_data_path, output_dir):
        """모든 Backbone 모델 테스트"""
        self.logger.info("Starting Backbone model testing...")
        
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 테스트 데이터 로드
        images, labels = self.load_test_data(test_data_path)
        
        # 테스트할 모델들
        models_to_test = [
            {
                'name': 'CellViT-SAM-H',
                'checkpoint': './checkpoints/classifier/CellViT-SAM-H-x40-AMP-002.pth'
            },
            {
                'name': 'CellViT-256',
                'checkpoint': './checkpoints/classifier/CellViT-256-x40-AMP.pth'
            },
            {
                'name': 'CellViT-Virchow',
                'checkpoint': './checkpoints/classifier/CellViT-Virchow-x40-AMP-001.pth'
            }
        ]
        
        # 각 모델 테스트
        for model_config in models_to_test:
            model_name = model_config['name']
            checkpoint_path = model_config['checkpoint']
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Testing {model_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                # 모델 로드
                model = self.load_backbone_model(model_name, checkpoint_path)
                
                # 첫 번째 이미지로 테스트
                test_image = images[0]
                image_tensor = self.preprocess_image(test_image)
                
                # 추론 실행
                outputs, inference_time = self.run_inference(model, image_tensor)
                
                # 출력 분석
                analysis = self.analyze_outputs(outputs, model_name)
                
                # 결과 저장
                self.results[model_name] = {
                    'analysis': analysis,
                    'inference_time': inference_time,
                    'model_size': sum(p.numel() for p in model.parameters())
                }
                
                # 시각화
                viz_path = output_dir / f"{model_name}_outputs.png"
                self.visualize_results(test_image, outputs, viz_path)
                
                # 상세 분석 출력
                self.logger.info(f"Model: {model_name}")
                self.logger.info(f"Inference Time: {inference_time:.3f}s")
                self.logger.info(f"Model Size: {self.results[model_name]['model_size']:,} parameters")
                self.logger.info("Output Analysis:")
                for key, value in analysis['output_shapes'].items():
                    self.logger.info(f"  {key}: {value}")
                
            except Exception as e:
                self.logger.error(f"Error testing {model_name}: {str(e)}")
                self.results[model_name] = {'error': str(e)}
        
        # 종합 결과 저장
        self.save_comprehensive_results(output_dir)
        
    def save_comprehensive_results(self, output_dir):
        """종합 결과 저장"""
        # 결과 요약
        summary = []
        for model_name, result in self.results.items():
            if 'error' not in result:
                summary.append({
                    'Model': model_name,
                    'Inference Time (s)': f"{result['inference_time']:.3f}",
                    'Model Size (M)': f"{result['model_size']/1e6:.1f}",
                    'Output Keys': ', '.join(result['analysis']['output_keys'])
                })
            else:
                summary.append({
                    'Model': model_name,
                    'Inference Time (s)': 'ERROR',
                    'Model Size (M)': 'ERROR',
                    'Output Keys': result['error']
                })
        
        # CSV로 저장
        df = pd.DataFrame(summary)
        df.to_csv(output_dir / 'backbone_test_results.csv', index=False)
        
        # 상세 결과를 YAML로 저장
        with open(output_dir / 'detailed_results.yaml', 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
        
        self.logger.info(f"\nResults saved to {output_dir}")
        self.logger.info("\nBackbone Test Summary:")
        print(df.to_string(index=False))

def main():
    """메인 함수"""
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 테스터 초기화
    tester = BackboneTester(device=device)
    
    # 테스트 실행
    test_data_path = "./test_database/training_database/Example-Detection/test"
    output_dir = "./backbone_test_results"
    
    tester.test_backbone_models(test_data_path, output_dir)

if __name__ == "__main__":
    main() 