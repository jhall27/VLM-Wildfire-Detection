# Wildfire UAV Dataset Documentation

## 1. Overview

**Project:** Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models

**Objective:** Document the combined dataset used for training and evaluating wildfire detection models.

**Date Created:** [INSERT DATE]

**Last Updated:** [INSERT DATE]

---

## 2. Dataset Sources

### 2.1 Source 1: AI For Mankind Data
- **Description:** [Describe the AI For Mankind dataset]
- **Download Link:** [Provide link if available]
- **Images:** [Count]
- **License:** [License information]

### 2.2 Source 2: Boreal Forest Fire Subset-C
- **Description:** Subset-C from the Boreal Forest Fire dataset used in the WACV paper
- **Download Link:** [Provide link if available]
- **Images:** [Count]
- **License:** [License information]

---

## 3. Combined Dataset Statistics

### 3.1 Image Counts

| Split | Count | Percentage |
|-------|-------|------------|
| Train | [NUMBER] | [PERCENTAGE]% |
| Test  | [NUMBER] | [PERCENTAGE]% |
| Valid | [NUMBER] | [PERCENTAGE]% |
| **Total** | **[TOTAL]** | **100%** |

### 3.2 Image Properties

| Property | Value |
|----------|-------|
| Format | [e.g., JPEG, PNG] |
| Average Resolution | [e.g., 1280x720] |
| Color Channels | RGB (3 channels) |
| Total File Size | [e.g., 50 GB] |

### 3.3 Label Format

| Property | Value |
|----------|-------|
| Label Type | [Pixel-wise masks / Bounding boxes / Polygons] |
| Format | [e.g., PNG, TXT] |
| Classes | Fire / Smoke / Background |
| Annotation Tool | [Tool used for annotation] |

---

## 4. Data Organization

### 4.1 Directory Structure

```
data/
├── images/
│   ├── train/          (X images)
│   ├── test/           (X images)
│   └── valid/          (X images)
├── labels/
│   ├── train/          (X labels)
│   ├── test/           (X labels)
│   └── valid/          (X labels)
├── manual_masks/       (Optional: ground truth masks)
└── sam_masks/          (Optional: SAM pseudo-labels)
```

### 4.2 Sample File Names

**Image:** `karkkila_DJI_0008_frame48.jpg`
**Label:** `karkkila_DJI_0008_frame48.png` (or `.txt`)

---

## 5. Data Exploration Results

### 5.1 Sample Images

[INSERT IMAGE: dataset_samples.png]

*Figure 1: Sample images and corresponding labels from train, test, and valid splits*

### 5.2 Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Background | [NUMBER] | [PERCENTAGE]% |
| Smoke | [NUMBER] | [PERCENTAGE]% |
| Fire | [NUMBER] | [PERCENTAGE]% |

**Note:** [Any class imbalance issues? Discuss here]

---

## 6. Data Quality Issues

### 6.1 Missing Files

- **Missing Labels:** [NUMBER] images are missing corresponding labels
- **Missing Images:** [NUMBER] labels are missing corresponding images
- **Affected Splits:** [Which splits are affected?]

### 6.2 Weak Annotations

- [Describe any weakly labeled samples]
- [Examples of ambiguous annotations]

### 6.3 Corrupted Files

- **Count:** [NUMBER]
- **Details:** [List corrupted files if few, or describe pattern]

### 6.4 Preprocessing Issues

- [Any images with unusual colors/brightness?]
- [Any extreme resolutions or aspect ratios?]
- [Any metadata issues?]

---

## 7. Data Preprocessing and Normalization

### 7.1 Image Preprocessing

- **Resizing:** [e.g., 1024x1024, or resize_to_fit=True]
- **Normalization:** [Mean/Std used for normalization]
- **Augmentation:** [Data augmentation techniques applied during training]

### 7.2 Label Preprocessing

- **Binarization:** [For binary segmentation: fire vs. non-fire]
- **Resizing:** [Labels resized to match images]
- **Encoding:** [How classes are encoded (0=background, 1=smoke, 2=fire, etc.)]

---

## 8. Data Loader Verification

### 8.1 Verification Steps Taken

- [x] Loaded sample images successfully
- [x] Verified label dimensions match image dimensions
- [x] Checked for missing files
- [ ] Verified preprocessing is correct
- [ ] Checked for memory leaks in data loader

### 8.2 Test Runs

**Test Case 1: Load single image with label**
```python
# Result: PASSED / FAILED
# Details: [Any issues encountered?]
```

**Test Case 2: Load batch of 16 images**
```python
# Result: PASSED / FAILED
# Details: [Any issues encountered?]
```

**Test Case 3: Data augmentation**
```python
# Result: PASSED / FAILED
# Details: [Any issues encountered?]
```

---

## 9. Dataset Splits and Usage

### 9.1 Training Set (Train Split)

- **Purpose:** Model training
- **Images:** [COUNT]
- **Characteristics:** [Any specific characteristics or domain?]

### 9.2 Validation Set (Valid Split)

- **Purpose:** Hyperparameter tuning and early stopping
- **Images:** [COUNT]
- **Characteristics:** [Any specific characteristics?]

### 9.3 Test Set (Test Split)

- **Purpose:** Final evaluation and results reporting
- **Images:** [COUNT]
- **Characteristics:** [Any specific characteristics?]

---

## 10. Known Limitations

1. [Limitation 1: e.g., limited night-time fire samples]
2. [Limitation 2: e.g., only UAV-based imagery, no ground cameras]
3. [Limitation 3: e.g., certain geographic regions under-represented]

---

## 11. Reproducing the Dataset

### 11.1 Download Instructions

```bash
# Step 1: Download AI For Mankind Data
# [Provide download command or link]

# Step 2: Download Boreal Forest Fire Subset-C
# [Provide download command or link]

# Step 3: Extract to the same directory
unzip ai_for_mankind_data.zip
unzip boreal_forest_fire_subset_c.zip

# Step 4: Organize into data/ folder
# [Provide organization script or instructions]
```

### 11.2 Verification

Run the data exploration notebook to verify dataset integrity:

```bash
jupyter notebook data_exploration.ipynb
```

---

## 12. References and Acknowledgments

- **Original Dataset Papers:**
  - [AI For Mankind Paper/Link]
  - [Boreal Forest Fire Paper: "Detecting Wildfires on UAVs with Real-time Segmentation..."]

- **Data Providers:**
  - AI For Mankind Team
  - Boreal Forest Fire Dataset Authors

---

## Appendix: Code for Data Exploration

See `data_exploration.ipynb` for code to reproduce all analysis and statistics.

### Quick Commands to Check Dataset

```bash
# Count train images
ls data/images/train | wc -l

# Count test images
ls data/images/test | wc -l

# Count valid images
ls data/images/valid | wc -l

# Check for missing labels
for img in data/images/train/*; do 
  base=$(basename "$img" .jpg); 
  if [ ! -f "data/labels/train/${base}.png" ]; then 
    echo "Missing label: $base"; 
  fi; 
done
```

---

**Document Status:** [Draft / In Progress / Complete]

**Prepared By:** [Your Name]

**Team Members:** [List team members]

**Last Reviewed:** [Date]
