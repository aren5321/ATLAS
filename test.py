"""
By default this script reads from data/test/.
Place the test data with the following structure:
data/test/
└── {task_id}/ (your task_id)
    ├── imagesTr/
    ├── labelsTr/ (only needed for segmentation/detection)
    └── labels.csv (only needed for classification/regression)
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import glob
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import project modules
from task_manager import TaskManager
from model_factory import MultiTaskModelFactory
from utils import evaluate, set_seed


# =========================================================================
# 1. Standalone test preprocessor (does not write into task_db, only builds DataFrame in memory)
# =========================================================================
class TestPreprocessor:
    def __init__(self, task_id, test_root="./data/test"):
        self.task_id = task_id
        self.root = os.path.join(test_root, task_id)
        self.img_dir = os.path.join(self.root, 'imagesTr')
        self.tm = TaskManager()

    def parse_voc_xml(self, xml_path, num_objects):
        """Reuse XML parsing logic"""
        results = {}
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs = root.findall('object')
            for i in range(num_objects):
                idx = i + 1
                if i < len(objs):
                    bbox = objs[i].find('bndbox')
                    results[f"x_min_{idx}"] = int(bbox.find('xmin').text)
                    results[f"y_min_{idx}"] = int(bbox.find('ymin').text)
                    results[f"x_max_{idx}"] = int(bbox.find('xmax').text)
                    results[f"y_max_{idx}"] = int(bbox.find('ymax').text)
                else:
                    results[f"x_min_{idx}"] = results[f"y_min_{idx}"] = 0
                    results[f"x_max_{idx}"] = results[f"y_max_{idx}"] = 0
        except:
            pass
        return results

    def run(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Test dataset path does not exist: {self.root}")

        cfg = self.tm.get_task_config(self.task_id)
        task_name = cfg['task_name'].lower()
        num_classes = cfg['num_classes']

        # 1. Load labels (if any)
        labels_lookup = {}
        if task_name in ['classification', 'regression']:
            label_csv = os.path.join(self.root, 'labels.csv')
            if not os.path.exists(label_csv):
                raise FileNotFoundError(f"For classification/regression tasks, labels.csv must be provided in test set: {label_csv}")

            # Use the same logic as preprocess.py to read labels
            df = pd.read_csv(label_csv)
            # Handle header differences between classification and regression
            if task_name == 'classification':
                # Be compatible with either 'label' or 'mask' column name
                key = 'label' if 'label' in df.columns else 'mask'
                labels_lookup = df.set_index('filename')[key].to_dict()
            elif task_name == 'regression':
                labels_lookup = df.set_index('filename').to_dict('index')

        # 2. Scan images
        img_list = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        if not img_list:
            raise FileNotFoundError(f"imagesTr is empty: {self.img_dir}")

        rows = []
        for img_p in img_list:
            fname = os.path.basename(img_p)
            bname = os.path.splitext(fname)[0]

            # Basic information
            row = {
                "image_path": img_p,  # keep absolute/relative path for Dataset reading
                "filename": fname,
                "task_name": task_name,
                "task_id": self.task_id,
                "num_classes": num_classes
            }

            # Fill labels
            if task_name == 'classification':
                if fname not in labels_lookup: continue  # skip if this image is not in CSV
                row["mask"] = labels_lookup[fname]  # reuse 'mask' field to store label

            elif task_name == 'regression':
                if fname not in labels_lookup: continue
                for i in range(1, num_classes + 1):
                    col = f"point_{i}_xy"
                    # Try to get from lookup, use default if missing
                    val = labels_lookup[fname].get(col, "[0, 0]")
                    row[col] = val

            elif task_name == 'segmentation':
                # Dynamically search for mask
                mask_search = glob.glob(os.path.join(self.root, 'labelsTr', bname + '.*'))
                if mask_search:
                    row["mask_path"] = mask_search[0]
                else:
                    # If there is no mask, Dice cannot be computed, skip this sample
                    print(f"[Warning] Skip sample without mask: {fname}")
                    continue

            elif task_name == 'detection':
                xml_p = os.path.join(self.root, 'labelsTr', bname + '.xml')
                if os.path.exists(xml_p):
                    row["mask_path"] = xml_p
                    row.update(self.parse_voc_xml(xml_p, num_classes))
                else:
                    continue

            rows.append(row)

        print(f"[*] Preprocessing finished: {self.task_id}, valid test samples: {len(rows)}")
        return pd.DataFrame(rows)


# =========================================================================
# 2. Standalone test Dataset (independent of data/train)
# =========================================================================
class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        task_name = row['task_name']

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare label / mask
        target = None

        if task_name == 'classification':
            target = int(row['mask'])  # 'mask' field stores label

        elif task_name == 'segmentation':
            mask_path = row['mask_path']
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros(image.shape[:2], dtype=np.uint8)
            # Resize logic is delegated to transform; just pass mask to albumentations
            pass

        elif task_name == 'regression':
            # Parse coordinates
            kpts = []
            for i in range(1, row['num_classes'] + 1):
                pt_str = row.get(f"point_{i}_xy", "[0, 0]")
                pt = eval(pt_str)  # [x, y]
                kpts.append(pt)
            target = np.array(kpts, dtype=np.float32)

        elif task_name == 'detection':
            boxes = []
            img_h, img_w = image.shape[:2]  # get true height and width of current image
            for i in range(1, row['num_classes'] + 1):
                # 1. Extract original coordinates
                xmin = row.get(f"x_min_{i}", 0)
                ymin = row.get(f"y_min_{i}", 0)
                xmax = row.get(f"x_max_{i}", 0)
                ymax = row.get(f"y_max_{i}", 0)

                # 2. Force clipping into image range to avoid Albumentations errors
                xmin = max(0, min(xmin, img_w - 1))
                ymin = max(0, min(ymin, img_h - 1))
                xmax = max(0, min(xmax, img_w))
                ymax = max(0, min(ymax, img_h))

                # 3. Simple filter for invalid boxes (ensure xmax > xmin)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax, 1])
            target = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)

        # Build input for transform
        data_pair = {"image": image}

        if task_name == 'segmentation':
            data_pair["mask"] = mask if 'mask' in locals() else np.zeros(image.shape[:2])
        elif task_name == 'detection':
            data_pair["bboxes"] = target[:, :4] if len(target) > 0 else []
            data_pair["labels"] = [1] * len(data_pair["bboxes"])
        elif task_name == 'regression':
            data_pair["keypoints"] = target

        # Apply transform
        if self.transform:
            augmented = self.transform(**data_pair)
            image = augmented['image']

            if task_name == 'segmentation':
                target = augmented['mask'].long()
            elif task_name == 'detection':
                boxes = augmented['bboxes']
                if len(boxes) > 0:
                    b = boxes[0]  # [xmin, ymin, xmax, ymax]

                    # --- Normalize to align with training logic in dataset.py ---
                    # Get image width and height after transform
                    h_new, w_new = image.shape[1], image.shape[2]

                    # Normalize coordinates
                    xmin = b[0] / w_new
                    ymin = b[1] / h_new
                    xmax = b[2] / w_new
                    ymax = b[3] / h_new

                    # Wrap as Tensor with shape (4,) to fit calculate_iou
                    target = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                else:
                    target = torch.zeros(4, dtype=torch.float32)

            elif task_name == 'regression':
                # Normalize
                h, w = augmented['image'].shape[1], augmented['image'].shape[2]
                kpts = augmented['keypoints']
                norm_kpts = [[x / w, y / h] for x, y in kpts]
                target = torch.tensor(norm_kpts, dtype=torch.float32).flatten()
            elif task_name == 'classification':
                target = torch.tensor(target, dtype=torch.long)

        return {
            "image": image,
            "label": target,
            "task_id": row['task_id']
        }


def main():
    parser = argparse.ArgumentParser(description="ATLAS standalone test script")
    parser.add_argument('--task_id', type=str, required=True, help="Task ID to test")
    parser.add_argument('--data_root', type=str, default="./data/test", help="Root directory of test data")
    args = parser.parse_args()

    # 1. Initialize manager
    tm = TaskManager()
    try:
        cfg = tm.get_task_config(args.task_id)
    except ValueError:
        print(f"[Error] Task {args.task_id} is not registered in task_manager, cannot load config.")
        return

    print(f"\n>>> Start testing pipeline: {args.task_id} ({cfg['task_name']})")

    # 2. Run standalone preprocessing
    try:
        preprocessor = TestPreprocessor(args.task_id, test_root=args.data_root)
        test_df = preprocessor.run()
    except Exception as e:
        print(f"[Error] Preprocessing failed: {e}")
        return

    if len(test_df) == 0:
        print("[Error] No valid test samples found.")
        return

    # 3. Prepare data augmentation (only Resize + Normalize)
    # Read resolution used during training; if missing, default to 512
    h = (cfg.get('resample_target', [512, 512])[0] // 32) * 32
    w = (cfg.get('resample_target', [512, 512])[1] // 32) * 32

    # Base transform list
    base_transforms = [
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]

    # Dynamically build Compose according to task type
    task_name = cfg['task_name'].lower()

    if task_name == 'regression':
        # Regression task: only add keypoint parameters
        val_transforms = A.Compose(
            base_transforms,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )
    elif task_name == 'detection':
        # Detection task: only add BBox parameters
        val_transforms = A.Compose(
            base_transforms,
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
    else:
        # Classification/segmentation tasks: no extra parameters needed
        val_transforms = A.Compose(base_transforms)

    # 4. Build Dataset and DataLoader
    test_ds = TestDataset(test_df, transform=val_transforms)
    # Note: reuse utils.multi_task_collate_fn to keep format consistent
    from utils import multi_task_collate_fn
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=4, collate_fn=multi_task_collate_fn)

    # 5. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model... (Device: {device})")

    # Only need to instantiate model branch for current task
    factory = MultiTaskModelFactory([{'task_id': args.task_id,
                                      'task_name': cfg['task_name'],
                                      'num_classes': cfg['num_classes']}])

    # model = factory.models[args.task_id]

    # Load weights
    weight_path = os.path.join("weights", f"{args.task_id}.pth")
    if not os.path.exists(weight_path):
        print(f"[Error] Weight file does not exist: {weight_path}")
        return

    # Use factory built-in method to load weights; it will automatically handle prefixes
    factory.load_weights_to_branch(args.task_id, weight_path)
    factory.to(device)
    factory.eval()

    # 6. Run evaluation
    print("[*] Running inference and metric computation...")

    # Pass the whole factory instead of a single model branch
    results_df = evaluate(factory, test_loader, device)

    # 7. Output results
    print("\n" + "=" * 40)
    print(f"   Test report: {args.task_id}")
    print("=" * 40)
    if not results_df.empty:
        print(results_df.to_string(index=False))

        # Optional: save results to local CSV for inspection
        out_csv = os.path.join(args.data_root, args.task_id, "test_results_metrics.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"\n[Done] Detailed metrics have been saved to: {out_csv}")
    else:
        print("No valid results generated.")


if __name__ == "__main__":
    main()