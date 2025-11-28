# EdgeYoloSamPipeline

Edge-optimized computer vision project focused on real-time object detection and fine-grained segmentation using a hybrid YOLO + lightweight SAM-style architecture. The system integrates quantization, structured pruning, and knowledge distillation to aggressively reduce model size, memory footprint, and inference latency, enabling high-accuracy performance on resource-constrained edge hardware.

## 🚀 Features

- **YOLO-based Object Detection**: Pre-trained YOLOv8 models for fast and accurate object detection
- **Model Quantization**: FP16 and INT8 quantization for reduced model size and faster inference
- **TensorRT Optimization**: Optimized models for NVIDIA GPUs with TensorRT engine export
- **COCO Dataset Support**: Full support for COCO dataset format with detection and segmentation tasks
- **Model Export**: Export models to ONNX, TensorRT, and PyTorch formats
- **Comprehensive Benchmarking**: Built-in benchmarking tools for model performance evaluation
- **Modular Architecture**: Clean, modular codebase with separation of concerns

## 📋 Requirements

- Python 3.12+
- CUDA-capable GPU (for TensorRT optimization)
- Conda (recommended) or pip for package management

## 🔧 Installation

### Using Conda (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd EdgeYoloSamPipeline
```

2. Create and activate the conda environment:
```bash
conda env create -f conda.yml
conda activate pytorch_yolosam
```

3. Install additional dependencies:
```bash
pip install ultralytics pycocotools
```

### Using pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics pycocotools numpy pandas matplotlib opencv-python datasets torchmetrics tqdm
```

## 📁 Project Structure

```
EdgeYoloSamPipeline/
├── configs/                 # Configuration files
│   ├── coco_yolo.yaml      # COCO dataset configuration
│   ├── yolo.yaml           # YOLO model configuration
│   ├── quantization.yaml   # Quantization settings
│   ├── distilation.yaml    # Knowledge distillation config
│   └── sam_light.yaml      # Lightweight SAM configuration
├── data/                    # Raw data directory (gitignored)
│   └── raw/                # Original COCO dataset files
├── datasets/                # Processed datasets
│   └── coco/               # COCO dataset in YOLO format
├── models/                  # Model checkpoints (gitignored)
│   ├── pretrained/         # Pre-trained models
│   └── final/              # Final optimized models
├── notebooks/               # Jupyter notebooks
│   ├── download_data.ipynb # Download COCO dataset
│   ├── data_preparation.ipynb  # Data preprocessing
│   └── train.ipynb         # Training workflow
├── runs/                    # Training runs and results (gitignored)
├── scripts/                 # Utility scripts
│   └── benchmark.py        # Model benchmarking script
├── src/                     # Source code
│   ├── compression/        # Model compression utilities
│   ├── data/               # Data loading and preprocessing
│   │   ├── datasets.py     # COCO dataset classes
│   │   ├── dataloaders.py  # DataLoader utilities
│   │   └── coco_to_yolo.py # COCO to YOLO format converter
│   ├── detection/          # Object detection models
│   │   └── yolo_baseline.py # YOLO baseline implementation
│   ├── pipelines/          # End-to-end pipelines
│   ├── segmentation/       # Segmentation models
│   └── utils/              # Utility functions
├── metrics/                 # Evaluation metrics (gitignored)
├── conda.yml               # Conda environment specification
└── README.md               # This file
```

## 🎯 Quick Start

### 1. Download Data

Use the `download_data.ipynb` notebook to download the COCO dataset:
- Training images (train2017.zip)
- Validation images (val2017.zip)
- Annotations (annotations_trainval2017.zip)

### 2. Prepare Data

Run `data_preparation.ipynb` to:
- Convert COCO annotations to YOLO format
- Create data loaders for training
- Verify data loading and visualization

### 3. Training

Follow the workflow in `train.ipynb`:

#### A. Pre-trained Baseline
```python
from ultralytics import YOLO

model = YOLO("models/pretrained/yolov8n.pt")

# Validate pre-trained model
model.val(
    data="coco.yaml", 
    imgsz=640, 
    batch=64,
    project="metrics",
    name="yolov8n_coco_pretrained"
)
```

#### B. Model Export and Quantization

**FP16 Quantization:**
```python
model = YOLO("models/pretrained/yolov8n.pt")

model.export(
    format="engine",
    half=True,
    imgsz=640,
    project="models/pretrained",
    name="yolov8n_fp16"
)
```

**INT8 Quantization:**
```python
model = YOLO("models/pretrained/yolov8n.pt")

model.export(
    format="engine",
    int8=True,
    data="coco.yaml",
    imgsz=640,
    project="models/pretrained",
    name="yolov8n_int8"
)
```

**ONNX Export:**
```python
model.export(
    format="onnx",
    imgsz=640,
    project="models/pretrained",
    name="yolov8n_pretrained_onnx"
)
```

### 4. Benchmarking

Run the benchmark script to compare model performance:
```bash
python scripts/benchmark.py
```

This will benchmark:
- FP32 PyTorch model
- FP16 TensorRT engine
- INT8 TensorRT engine

## 📊 Model Formats

The project supports multiple model formats:

- **PyTorch (.pt)**: Original PyTorch model format
- **ONNX (.onnx)**: ONNX format for cross-platform deployment
- **TensorRT Engine (.engine)**: Optimized for NVIDIA GPUs
  - FP16: Reduced precision with minimal accuracy loss
  - INT8: Maximum compression with calibration data

## 🔬 Data Processing

### COCO Dataset Support

The project includes a custom `CocoDetectionSegmentation` dataset class that handles:
- Image loading and preprocessing
- Bounding box annotations
- Instance segmentation masks
- Automatic resizing and normalization

Example usage:
```python
from src.data.datasets import CocoDetectionSegmentation

dataset = CocoDetectionSegmentation(
    img_dir="data/raw/images/train2017",
    ann_path="data/raw/annotations/instances_train2017.json",
    image_size=640
)

image, boxes, masks = dataset[0]
```

### DataLoader

Create optimized data loaders:
```python
from src.data.dataloaders import create_dataloader

loader = create_dataloader(
    img_dir="data/raw/images/train2017",
    ann_path="data/raw/annotations/instances_train2017.json",
    batch_size=8,
    num_workers=4
)
```

## ⚙️ Configuration

Configuration files are stored in the `configs/` directory:

- **coco_yolo.yaml**: COCO dataset configuration with 80 classes
- **yolo.yaml**: YOLO model architecture settings
- **quantization.yaml**: Quantization parameters
- **distilation.yaml**: Knowledge distillation settings
- **sam_light.yaml**: Lightweight SAM model configuration

## 📈 Evaluation Metrics

Models are evaluated using standard COCO metrics:
- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Mean Average Precision averaged over IoU 0.50:0.95
- **Precision (P)**: Precision metric
- **Recall (R)**: Recall metric
- **F1-Score**: Harmonic mean of precision and recall

Visualization outputs include:
- Confusion matrices
- PR curves
- F1 curves
- Prediction visualizations

## 🛠️ Advanced Usage

### Custom Training

For custom training configurations, modify the training notebook or create a new script:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="configs/coco_yolo.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/train"
)
```

### Model Compression Pipeline

The project supports various compression techniques:
1. **Quantization**: FP16 and INT8
2. **Pruning**: Channel and layer-level pruning (planned)
3. **Knowledge Distillation**: Teacher-student training (planned)

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training/evaluation
2. **TensorRT Export Fails**: Ensure CUDA and TensorRT are properly installed
3. **Data Loading Slow**: Increase `num_workers` in DataLoader or use cached datasets

## 📝 Notes

- Models and data directories are gitignored by default
- Training results are saved in the `runs/` directory
- Pre-trained models should be downloaded to `models/pretrained/`
- Dataset should be placed in `data/raw/` or `datasets/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is part of an Advanced Computer Vision course. Please refer to the respective licenses of:
- Ultralytics YOLO
- COCO Dataset
- PyTorch

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- PyTorch team

## 📚 References

- YOLOv8: [Ultralytics Documentation](https://docs.ultralytics.com/)
- COCO Dataset: [COCO Documentation](https://cocodataset.org/#home)
- TensorRT: [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)

---

For questions or issues, please open an issue on the repository.
