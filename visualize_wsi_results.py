#!/usr/bin/env python3
"""
WSI 썸네일 위에 CellViT Backbone 결과 시각화
세포 중심점과 bbox를 WSI 전체 이미지 위에 표시
"""

import json
import openslide
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os

def get_thumbnail(svs_path, scale_factor=32):
    """WSI 썸네일 추출"""
    slide = openslide.OpenSlide(svs_path)
    thumbnail = slide.get_thumbnail((slide.dimensions[0] // scale_factor, slide.dimensions[1] // scale_factor))
    return slide, thumbnail

def load_cells(json_path):
    """Backbone 결과 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['cells']  # cells 배열만 반환

def visualize_cells_on_wsi(slide, thumbnail, cells, save_path):
    """WSI 위에 세포 검출 결과 시각화"""
    scale_x = thumbnail.size[0] / slide.dimensions[0]
    scale_y = thumbnail.size[1] / slide.dimensions[1]
    thumb_np = np.array(thumbnail)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(thumb_np)

    # 세포 타입별 색상 매핑
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for cell in cells:
        # 중심점 - centroid는 [y, x] 형태
        centroid_y = float(cell['centroid'][0])
        centroid_x = float(cell['centroid'][1])
        offset_y = float(cell['offset_global'][0])
        offset_x = float(cell['offset_global'][1])
        
        # global 좌표 계산
        y_global = centroid_y + offset_y
        x_global = centroid_x + offset_x
        
        # 썸네일 좌표로 변환
        x_thumb = x_global * scale_x
        y_thumb = y_global * scale_y
        
        cell_type = cell.get('type', 0)
        color = colors[cell_type % len(colors)]
        
        # 중심점 표시
        ax.scatter(x_thumb, y_thumb, c=color, s=3, alpha=0.8)
        
        # bbox - bbox는 [[x1, y1], [x2, y2]] 형태
        bbox = cell['bbox']
        x1, y1 = float(bbox[0][0]), float(bbox[0][1])
        x2, y2 = float(bbox[1][0]), float(bbox[1][1])
        
        # global 좌표로 변환
        x1g = (x1 + offset_x) * scale_x
        y1g = (y1 + offset_y) * scale_y
        x2g = (x2 + offset_x) * scale_x
        y2g = (y2 + offset_y) * scale_y
        
        # bbox 그리기
        rect = patches.Rectangle((x1g, y1g), x2g-x1g, y2g-y1g, 
                               linewidth=0.3, edgecolor=color, 
                               facecolor='none', alpha=0.6)
        ax.add_patch(rect)

    plt.title('CellViT Backbone Cell Detection Results\n(Centroid + BBox)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"시각화 결과가 저장되었습니다: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 파일 경로 설정
    svs_path = './test_database/x40_svs/JP2K-33003-2.svs'
    json_path = './backbone_test_results/JP2K-33003-2_cells.json'
    save_path = './backbone_test_results/JP2K-33003-2_cells_viz_bbox.png'

    print("WSI 썸네일 추출 중...")
    slide, thumbnail = get_thumbnail(svs_path)
    print(f"썸네일 크기: {thumbnail.size}")
    
    print("Backbone 결과 로드 중...")
    cells = load_cells(json_path)
    print(f"검출된 세포 수: {len(cells)}")
    
    print("시각화 생성 중...")
    visualize_cells_on_wsi(slide, thumbnail, cells, save_path) 