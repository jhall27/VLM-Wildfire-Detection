"""
Dataset Loader for Clemson Wildfire VLM
Handles loading images and labels with proper validation and preprocessing
"""

import os
import cv2 #OperCV
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
from xml.etree import ElementTree as ET

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WildfireDataset(Dataset):
    """
    Clean dataset loader for Wildfire VLM

    Usage:
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (768, 768), #best size since images are 2048 x 1024
        normalize: bool = True,
        skip_missing: bool = True,
        verbose: bool = True, #When true, prints all the information
    ):
        """
        Initialize the dataset loader.
        
        Args:
            image_dir: Path to images directory (contains train/test/valid subdirs)
            label_dir: Path to labels directory (contains train/test/valid subdirs)
            split: Dataset split ('train', 'test', or 'valid')
            image_size: Target image size as (height, width)
            normalize: Whether to normalize images to [0, 1]
            skip_missing: Skip images with missing labels (True) or raise error (False)
            verbose: Print detailed logging information
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.skip_missing = skip_missing
        self.verbose = verbose

        # Validate directories exist
        self._validate_directories()
        
        # Load dataset
        self.samples = self._load_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found for {split} split!")
        
        logger.info(f"Dataset initialized: {len(self.samples)} valid samples for {split} split")
    
    def _validate_directories(self):
        """Validate that required directories exist."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        split_img_dir = self.image_dir / self.split
        split_lbl_dir = self.label_dir / self.split
        
        if not split_img_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_img_dir}")
        if not split_lbl_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_lbl_dir}")
    
    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load and validate all image-label pairs."""
        samples = []
        missing_count = 0
        error_count = 0
        
        # Get all images
        image_files = self._get_image_files()
        
        if self.verbose:
            logger.info(f"\nLoading {self.split} split:")
            logger.info(f"  Found {len(image_files)} images")
        
        for img_path in sorted(image_files):
            try:
                # Find corresponding label
                label_path = self._find_label(img_path)
                
                if label_path is None:
                    missing_count += 1
                    if self.skip_missing:
                        if self.verbose:
                            logger.warning(f"Skipping: {img_path.name} (no label found)")
                        continue
                    else:
                        raise FileNotFoundError(f"No label found for {img_path.name}")
                
                # Quick validation that image can be loaded
                img = cv2.imread(str(img_path)) #NumPy array with shape (1080, 2048, 3)
                if img is None:
                    error_count += 1
                    if self.verbose:
                        logger.error(f"Failed to load image: {img_path.name}")
                    continue
                
                # Add to samples
                samples.append({
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "image_name": img_path.stem,
                })
                
            except Exception as e:
                error_count += 1
                if self.verbose:
                    logger.error(f"  ✗ Error processing {img_path.name}: {e}")
                continue
        
        # Log statistics
        if self.verbose:
            logger.info(f"\n Valid samples: {len(samples)}")
            logger.info(f"Missing labels: {missing_count}")
            logger.info(f"Corrupted files: {error_count}")
        
        return samples
    
    def _get_image_files(self) -> List[Path]:
        """Get all image files from the split directory."""
        image_suf = ['.jpg', '.jpeg']
        images = []
        
        split_dir = self.image_dir / self.split
        
        for ext in image_suf:
            images.extend(split_dir.glob(f"*{ext}"))
        
        return images
    
    def _find_label(self, img_path: Path) -> Optional[Path]:
        """Find corresponding label file (supports PNG, TXT, XML)."""
        label_base = img_path.stem
        split_dir = self.label_dir / self.split
        
        # Check for labels in order of preference
        for ext in ['.txt', '.xml']:
            label_path = split_dir / f"{label_base}{ext}"
            if label_path.exists():
                return label_path
        
        return None
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image and return as RGB array."""

        #Format is BGR (default of OpenCV)
        img = cv2.imread(img_path) #NumPy array with shape (1080, 2048, 3)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size (less data/still kind of precise)
        # Resize takes (width, height)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Normalize so pixel is between 0 - 1.0
        img = img.astype(np.float32) / 255.0
        
        return img

    def _load_label(self, label_path: str) -> np.ndarray:
        """Load (normalized bounding boxes) YOLO"""
        label_path = Path(label_path)
        mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)

        if label_path.suffix == '.txt':
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            cx = float(parts[1])  # Normalized center X
                            cy = float(parts[2])  # Normalized center Y
                            w = float(parts[3])  # Normalized width
                            h = float(parts[4])  # Normalized height

                            # Convert to pixel coordinates
                            x1 = int((cx - w / 2) * self.image_size[1])
                            y1 = int((cy - h / 2) * self.image_size[0])
                            x2 = int((cx + w / 2) * self.image_size[1])
                            y2 = int((cy + h / 2) * self.image_size[0])

                            # Clip to image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(self.image_size[1], x2)
                            y2 = min(self.image_size[0], y2)

                            # Draw rectangle on mask
                            if x2 > x1 and y2 > y1:
                                mask[y1:y2, x1:x2] = 1
                        except:
                            continue
            except Exception as e:
                logger.warning(f"Error reading TXT label {label_path}: {e}")

        elif label_path.suffix == '.xml':
            # XML format
            try:
                tree = ET.parse(label_path)
                root = tree.getroot()

                # Parse objects (adjust tag names based on your XML format)
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        try:
                            x1 = int(float(bndbox.find('xmin').text))
                            y1 = int(float(bndbox.find('ymin').text))
                            x2 = int(float(bndbox.find('xmax').text))
                            y2 = int(float(bndbox.find('ymax').text))

                            # Clip to image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(self.image_size[1], x2)
                            y2 = min(self.image_size[0], y2)

                            # Draw rectangle on mask
                            if x2 > x1 and y2 > y1:
                                mask[y1:y2, x1:x2] = 1
                        except:
                            continue
            except Exception as e:
                logger.warning(f"Error reading XML label {label_path}: {e}")

        else:
            logger.warning(f"Unsupported label format: {label_path.suffix}")

        return mask
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary with:
            - 'image': torch.Tensor of shape (3, H, W), values in [0, 1]
            - 'mask': torch.Tensor of shape (1, H, W), values in {0, 1}
            - 'name': image filename
        """
        sample = self.samples[idx]
        
        # Load image and label
        image = self._load_image(sample['image_path'])
        mask = self._load_label(sample['label_path'])
        
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return {
            'image': image,
            'mask': mask,
            'name': sample['image_name'],
        }


def create_dataloaders(
    image_dir: str,
    label_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (1024, 1024),
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/test/valid dataloaders.
    
    Args:
        image_dir: Path to images directory
        label_dir: Path to labels directory
        batch_size: Batch size
        image_size: Target image size (H, W)
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Dictionary with 'train', 'test', 'valid' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'test', 'valid']:
        dataset = WildfireDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            split=split,
            image_size=image_size,
            normalize=True,
            skip_missing=True,
            verbose=True,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle_train if split == 'train' else False),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
        
        dataloaders[split] = dataloader
        logger.info(f"  → {split} dataloader: {len(dataloader)} batches\n")
    
    return dataloaders


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset loader
    logger.info("="*60)
    logger.info("WILDFIRE DATASET LOADER - TEST")
    logger.info("="*60 + "\n")
    
    # Create datasets
    # Num_workers change to 4+ for production
    dataloaders = create_dataloaders(
        image_dir="data/images",
        label_dir="data/labels",
        batch_size=4,
        image_size=(768, 768),
        num_workers=0,
    )
    
    # Test loading a batch
    logger.info("Testing batch loading...")
    train_loader = dataloaders['train']
    
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"\nBatch {batch_idx}:")
        logger.info(f"  Image shape: {batch['image'].shape}")
        logger.info(f"  Mask shape: {batch['mask'].shape}")
        logger.info(f"  Image dtype: {batch['image'].dtype}")
        logger.info(f"  Mask dtype: {batch['mask'].dtype}")
        logger.info(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        logger.info(f"  Mask range: [{batch['mask'].min():.0f}, {batch['mask'].max():.0f}]")
        logger.info(f"  Sample names: {batch['name']}")
        
        # Only test first batch
        break
    
    logger.info(" Dataset loader test passed!")