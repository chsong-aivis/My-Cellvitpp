import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from collections import Counter
import json
import openslide
import matplotlib.patches as patches

# Define color map for each class
CLASS_COLORS = {
    '0': (255, 0, 0),    # Red
    '1+': (0, 255, 0),   # Green
    '2+': (0, 0, 255),   # Blue
    '3+': (255, 255, 0), # Yellow
    'nt': (255, 0, 255)  # Magenta
}

def add_legend(img, class_counts, title="", pos_y=30):
    """Add color legend and class distribution to image"""
    img_with_legend = img.copy()
    
    # Calculate percentages
    total = sum(class_counts.values())
    percentages = {k: (v/total)*100 for k, v in class_counts.items()}
    
    # Add title
    cv2.putText(img_with_legend, title, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add color squares and percentages for each class
    x = 10
    for cls, count in class_counts.items():
        color = CLASS_COLORS.get(str(cls), (128, 128, 128))
        # Draw color square
        cv2.rectangle(img_with_legend, (x, pos_y), (x+15, pos_y+15), color, -1)
        # Add text
        text = f"{cls}: {percentages[cls]:.1f}%"
        cv2.putText(img_with_legend, text, (x+20, pos_y+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        x += 100
    
    return img_with_legend

def draw_points_on_image(img, points, labels, title=""):
    """Draw points with class-specific colors and labels on image"""
    img_with_points = img.copy()
    
    # Count instances of each class
    class_counts = Counter(labels)
    
    # Add legend with class distribution
    img_with_points = add_legend(img_with_points, class_counts, title)
    
    # Draw points with class-specific colors
    for point, label in zip(points, labels):
        x, y = map(int, point)
        color = CLASS_COLORS.get(str(label), (128, 128, 128))
        cv2.circle(img_with_points, (x, y), 3, color, -1)
        # Only add label text if it's not '0' to reduce clutter
        if str(label) != '0':
            cv2.putText(img_with_points, str(label), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return img_with_points

def load_image_and_points(image_path, label_path):
    """Load image and corresponding points with labels"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Read CSV file containing point information
    df = pd.read_csv(label_path, header=None, names=['x', 'y', 'class'])
    points = df[['x', 'y']].values
    labels = df['class'].values
        
    return img, points, labels

def visualize_results(val_dataset_path, inference_results_path1, inference_results_path2, output_path, num_samples=10):
    # Read inference results
    inference_df1 = pd.read_csv(inference_results_path1)
    inference_df2 = pd.read_csv(inference_results_path2)
    
    # Get unique image names (use the same images for both models)
    unique_images = inference_df1['Image'].unique()[:num_samples]
    
    # Create figure with 2x5 subplots
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.ravel()
    
    for idx, img_name in enumerate(unique_images):
        try:
            # Load original image and ground truth
            img_path = os.path.join(val_dataset_path, 'images', f'{img_name}.png')
            label_path = os.path.join(val_dataset_path, 'labels', f'{img_name}.csv')
            
            img, gt_points, gt_labels = load_image_and_points(img_path, label_path)
            
            # Get predictions for this image from both models
            img_preds1 = inference_df1[inference_df1['Image'] == img_name]
            pred_labels1 = img_preds1['Prediction'].tolist()
            
            img_preds2 = inference_df2[inference_df2['Image'] == img_name]
            pred_labels2 = img_preds2['Prediction'].tolist()
            
            # Draw ground truth points with class-specific colors
            gt_img = draw_points_on_image(img, gt_points, gt_labels, "Ground Truth")
            
            # Draw prediction points with class-specific colors for both models
            pred_img1 = draw_points_on_image(img.copy(), gt_points, pred_labels1, "No Weights")
            pred_img2 = draw_points_on_image(img.copy(), gt_points, pred_labels2, "With Weights")
            
            # Create side-by-side comparison of all three
            combined_img = np.hstack([gt_img, pred_img1, pred_img2])
            
            # Display in subplot
            axes[idx].imshow(combined_img)
            axes[idx].set_title(f'Image: {img_name}')
            axes[idx].axis('off')
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            axes[idx].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 1. WSI 썸네일 추출
def get_thumbnail(svs_path, scale_factor=32):
    slide = openslide.OpenSlide(svs_path)
    thumbnail = slide.get_thumbnail((slide.dimensions[0] // scale_factor, slide.dimensions[1] // scale_factor))
    return slide, thumbnail

# 2. backbone 결과 로드
def load_cells(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['cells']  # cells 배열만 반환

# 3. 시각화 함수
def visualize_cells_on_wsi(slide, thumbnail, cells, save_path):
    scale_x = thumbnail.size[0] / slide.dimensions[0]
    scale_y = thumbnail.size[1] / slide.dimensions[1]
    thumb_np = np.array(thumbnail)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(thumb_np)

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
        ax.scatter(x_thumb, y_thumb, c=f'C{cell_type%10}', s=5, alpha=0.7)
        
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
                               linewidth=0.5, edgecolor=f'C{cell_type%10}', 
                               facecolor='none', alpha=0.5)
        ax.add_patch(rect)

    plt.title('CellViT Backbone Cell Detection Results (Centroid + BBox)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

if __name__ == "__main__":
    val_dataset_path = "Validation_Dataset"
    inference_results_path1 = "inference_results/inference_results.csv"  # No weights
    inference_results_path2 = "inference_results_weighted/inference_results.csv"  # With weights
    output_path = "inference_results/model_comparison.png"
    
    visualize_results(val_dataset_path, inference_results_path1, inference_results_path2, output_path)
    print(f"Visualization saved to {output_path}")

    svs_path = './test_database/x40_svs/JP2K-33003-2.svs'
    json_path = './backbone_test_results/JP2K-33003-2_cells.json'
    save_path = './backbone_test_results/JP2K-33003-2_cells_viz_bbox.png'

    slide, thumbnail = get_thumbnail(svs_path)
    cells = load_cells(json_path)
    visualize_cells_on_wsi(slide, thumbnail, cells, save_path) 