import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from collections import Counter

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

if __name__ == "__main__":
    val_dataset_path = "Validation_Dataset"
    inference_results_path1 = "inference_results/inference_results.csv"  # No weights
    inference_results_path2 = "inference_results_weighted/inference_results.csv"  # With weights
    output_path = "inference_results/model_comparison.png"
    
    visualize_results(val_dataset_path, inference_results_path1, inference_results_path2, output_path)
    print(f"Visualization saved to {output_path}") 