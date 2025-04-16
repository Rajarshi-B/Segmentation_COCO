# assignment1_dataset_prep/README.md


# 🧠 Image Segmentation Assignment

This project demonstrates a complete image segmentation workflow using COCO-style datasets. It includes dataset preparation (mask generation), training semantic segmentation models (UNet and SegFormer), handling edge cases, and ensuring clean experimentation and reproducibility via WandB.

---

## 📁 Directory Structure
```
.
├── A1.ipynb                        # Notebook for dataset preparation (Task 1)
├── A2.ipynb                        # Notebook for UNet model training (Task 2)
├── data/
│   ├── raw/                        # Original COCO images and annotations
│   └── processed/
│       ├── images/                # Resized RGB images (256x256)
│       └── masks/                 # Processed multi-class masks (category_id values)
├── report.md                      # Summary and results (to be included)
├── requirements.in                # Dependency file for uv/pip setup
└── README.md                      # This file
```

---

## ✅ Task 1: Dataset Preparation (see `A1.ipynb`)

### 🗂️ Dataset
- Dataset used: **COCO 2017 validation set (val2017)**
- Total images processed: **6,000**, resized to **256×256**

### ⚙️ Processing Pipeline
- Parsed COCO annotations using `pycocotools.COCO`
- Generated masks where pixel values represent object `category_id`
- Implemented error handling for:
  - Missing or invalid `segmentation` fields
  - Empty polygons or malformed annotation entries
  - RLE-formatted masks (ignored as only polygons are supported)
- Output masks are saved as **single-channel PNGs** with `uint8` values
- **Note**: `category_id` remapping was deferred to training phase for flexibility

### 📊 Visual Examples (replace paths with actual images)
- ✅ Mask after filtering invalid annotations: `![](assets/sample_mask_filtered.jpg)`

---

## ✅ Task 2: Model Training

### 🧠 Models Explored
- ✅ `UNet` using `segmentation_models_pytorch` (see `A2.ipynb`)
  - **Successfully trains and logs metrics**
  - Validated on preprocessed COCO dataset with class index remapping
- ⚠️ `SegFormer` from HuggingFace Transformers
  - **Currently throws a CUDA device-side assert error** during training
  - The likely cause is label index mismatch or dynamic shape incompatibility
  - Under investigation and not included in final evaluation

### 🏗️ Training Configuration (for UNet)
- Loss function: `CrossEntropyLoss`
- Optimizer: `AdamW`, LR = `5e-5`
- Batch size: `4`, Epochs: `10`
- Label remapping from COCO `category_id → class index` handled during data loading
- Dtype and range validation enforced to avoid training errors

### 📊 Logging and Monitoring
- ✅ **WandB** used for tracking all experiments:
  - Loss per epoch, validation performance
  - Model configuration and training metadata
- Visual prediction logging to be added in final version
- `![](assets/WanDB.jpg)`

### 💾 Model Checkpoint
- Best UNet model saved as `best_model.pth` when validation loss improved

---

## 📥 Environment Setup
Install all dependencies using `uv`:
```bash
uv venv
uv pip install -r requirements.in
```

---

## 📌 Notes and Implementation Nuances
- **UNet training is complete and error-free**
- **SegFormer still has known issues** with mask shape/label handling
- All masks are single-channel PNGs with class indices (not one-hot)
- `category_id` remapping is mandatory for COCO masks, handled at dataset level
- Augmentation and normalization handled via `albumentations`

---

## 📎 Future Work / Enhancements
- 🔍 Debug SegFormer CUDA assert and fix label mapping/shaping issues
- 🧪 Generate overlay masks and prediction visualizations
- 📈 Compare UNet and SegFormer (once working) quantitatively
- 📤 Export best models to WandB/HuggingFace Hub

---
Note: This Readme was made with the assistance of LLM.

## 📝 License
MIT License
