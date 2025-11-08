"""
Phase 1 Task 1.2: Vision-Based Obstacle Detection
- Train YOLOv8/YOLOv5 for obstacle detection on AirSim data
- Input: RGB image
- Output: Bounding boxes + confidence scores
- Success Criteria: >85% mAP@50 on test set
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil

def prepare_yolo_dataset(data_dir="datasets/manual_collection"):
    """Prepare AirSim data for YOLOv8 training format"""
    data_path = Path(data_dir)
    
    # YOLO format: dataset/images/train/ and dataset/labels/train/
    yolo_dir = Path("datasets/yolo_obstacle_detection")
    
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Copy images
        src_images = data_path / split / "rgb"
        dst_images = yolo_dir / "images" / split
        
        if src_images.exists():
            for img_file in src_images.glob("*.png"):
                shutil.copy2(img_file, dst_images / img_file.name)
        
        # Create labels from segmentation masks
        src_seg = data_path / split / "segmentation"
        if src_seg.exists():
            create_labels_from_segmentation(src_seg, yolo_dir / "labels" / split)
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(yolo_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,  # Number of classes
        'names': ['obstacle', 'safe']  # Class names
    }
    
    with open(yolo_dir / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    print(f"[OK] YOLO dataset prepared at: {yolo_dir}")
    return yolo_dir / "dataset.yaml"

def create_labels_from_segmentation(seg_dir, label_dir):
    """Convert segmentation masks to YOLO format labels"""
    # Simplified: convert segmentation to bounding boxes
    # In practice, use proper segmentation-to-bbox conversion
    import cv2
    import numpy as np
    
    for seg_file in seg_dir.glob("*.png"):
        seg_img = cv2.imread(str(seg_file))
        h, w = seg_img.shape[:2]
        
        # Find obstacle regions (non-ground pixels)
        gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        # Threshold to find obstacles (simplified)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to YOLO format: class_id x_center y_center width height (normalized)
        label_file = label_dir / (seg_file.stem + ".txt")
        with open(label_file, 'w') as f:
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small regions
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    # Normalize
                    x_center = (x + w_box / 2) / w
                    y_center = (y + h_box / 2) / h
                    width_norm = w_box / w
                    height_norm = h_box / h
                    # Class 0 = obstacle
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

def train_yolov8(dataset_yaml, epochs=100, imgsz=640):
    """Train YOLOv8 model"""
    print("=" * 60)
    print("TRAINING YOLOv8 FOR OBSTACLE DETECTION")
    print("=" * 60)
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model for faster training
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0,  # GPU
        project='models/yolov8_obstacle',
        name='obstacle_detector',
        patience=20,  # Early stopping
        save=True,
        plots=True
    )
    
    print("[OK] Training complete!")
    return model

def evaluate_model(model, dataset_yaml):
    """Evaluate trained model"""
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    # Evaluate on validation set
    metrics = model.val(data=str(dataset_yaml))
    
    print(f"\n[RESULTS]")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Success criteria: >85% mAP@50
    if metrics.box.map50 >= 0.85:
        print("[SUCCESS] Model meets target (>85% mAP@50)!")
    else:
        print("[WARNING] Model below target, consider more training or data")
    
    return metrics

def main():
    # Step 1: Prepare dataset
    print("[1/3] Preparing YOLO dataset...")
    dataset_yaml = prepare_yolo_dataset()
    
    # Step 2: Train model
    print("\n[2/3] Training YOLOv8...")
    model = train_yolov8(dataset_yaml, epochs=100)
    
    # Step 3: Evaluate
    print("\n[3/3] Evaluating model...")
    metrics = evaluate_model(model, dataset_yaml)
    
    print("\n" + "=" * 60)
    print("PHASE 1 TASK 1.2 COMPLETE")
    print("=" * 60)
    print(f"Model saved to: models/yolov8_obstacle/obstacle_detector/")
    print(f"mAP@50: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    main()
