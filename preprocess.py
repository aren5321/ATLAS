"""
Usage
Process a single task (most common):   python preprocess.py --task_id FUGC
Batch process all tasks:               python preprocess.py

Dataset_Root/
├── imagesTr/           # training images
├── labelsTr/           # (segmentation/detection only) mask images or XML
├── labels.csv          # (classification/regression only) labels/coordinates for all images

A. Classification example
filename,label
img_001.jpg,0
img_002.jpg,1

B. Regression example
filename,point_1_xy,point_2_xy
img_001.png,"[239, 188]","[401, 102]"
img_002.png,"[250, 170]","[390, 90]"
"""
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
from task_manager import TaskManager


class AtlasPreprocessor:
    def __init__(self, data_root="./data/train", save_csv_dir="./data/train/csv_files"):
        # Lock project root: assume script runs at project root, or derive via this method
        self.project_root = os.getcwd()
        self.data_root = data_root
        self.save_csv_dir = save_csv_dir
        self.tm = TaskManager()
        os.makedirs(self.save_csv_dir, exist_ok=True)

    def get_rel_path(self, target_path):
        """Convert path to one relative to project root"""
        # Use relpath to ensure unified format: data/train/task_id/...
        return os.path.relpath(target_path, self.project_root)

    def parse_voc_xml_multi(self, xml_path, num_objects):
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
        except Exception as e:
            print(f"  [Error] XML parsing failed: {e}")
        return results

    def process_task(self, task_id):
        print(f"\n>>> Preprocessing task: {task_id}")
        try:
            cfg = self.tm.get_task_config(task_id)
        except ValueError:
            print(f"  [Skip] {task_id} is not registered")
            return

        task_name = cfg['task_name'].lower()
        num_classes = cfg['num_classes']
        task_dir = os.path.join(self.data_root, task_id)
        img_dir = os.path.join(task_dir, 'imagesTr')

        labels_lookup = {}
        if task_name in ['classification', 'regression']:
            label_csv = os.path.join(task_dir, 'labels.csv')
            # --- Additional check: duplicated filenames ---
            df_labels = pd.read_csv(label_csv)
            if df_labels['filename'].duplicated().any():
                dupes = df_labels[df_labels['filename'].duplicated()]['filename'].unique().tolist()
                raise ValueError(f"\n[ERROR] Task {task_id} labels.csv has duplicated filenames: {dupes[:3]}")

            labels_lookup = df_labels.set_index('filename').to_dict('index')

        img_list = sorted(glob(os.path.join(img_dir, '*')))
        # --- Use raise to hard-stop when missing images ---
        if not img_list:
            error_msg = (f"\n[CRITICAL ERROR] Task '{task_id}' preprocessing failed!\n"
                         f"Reason: imagesTr folder is empty or contains no image files.\n"
                         f"Check path: {os.path.abspath(img_dir)}")
            # Raise exception so program stops immediately and prints stack trace
            raise FileNotFoundError(error_msg)
        rows, image_shapes = [], []

        for img_p in tqdm(img_list, desc=f"Scanning"):
            fname = os.path.basename(img_p)
            bname = os.path.splitext(fname)[0]

            img = cv2.imread(img_p)
            # --- Extra check ---
            if img is None:
                # File is not a valid image (maybe txt, corrupted file, etc.)
                print(f"  [Skip] Failed to read file (possibly non-image): {os.path.basename(img_p)}")
                continue
            h, w = img.shape[:2]
            image_shapes.append((h, w))

            # --- Store path relative to project root ---
            rel_img_path = self.get_rel_path(img_p)

            row = {
                "image_path": rel_img_path,
                "height": h, "width": w,
                "task_name": task_name, "num_classes": num_classes, "task_id": task_id
            }

            if task_name == 'classification':
                row["mask"] = labels_lookup[fname]['label']
            elif task_name == 'regression':
                for i in range(1, num_classes + 1):
                    col = f"point_{i}_xy"
                    row[col] = labels_lookup[fname].get(col, "[0, 0]")
            elif task_name == 'segmentation':
                # Dynamic search: all possible suffixes under labelsTr with same basename
                mask_search = glob(os.path.join(task_dir, 'labelsTr', bname + '.*'))

                if not mask_search:
                    # If not found for any suffix, raise error and stop
                    raise FileNotFoundError(f"\n[ERROR] Segmentation task '{task_id}' missing mask file: "
                                            f"no image named '{bname}' found in labelsTr.")

                # Take the first match (usually only one)
                mask_p = mask_search[0]
                row["mask_path"] = self.get_rel_path(mask_p)
            elif task_name == 'detection':
                xml_p = os.path.join(task_dir, 'labelsTr', bname + '.xml')
                row["mask_path"] = self.get_rel_path(xml_p)
                row.update(self.parse_voc_xml_multi(xml_p, num_classes))

            rows.append(row)

        if image_shapes:
            shapes = np.array(image_shapes)
            fp_median = [int(np.median(shapes[:, 0])), int(np.median(shapes[:, 1]))]
            num_samples= len(rows)
            self.tm.db[task_id].update({"resample_target": fp_median, "status": "ready", "train_samples": num_samples})
            self.tm.save()

            pd.DataFrame(rows).to_csv(os.path.join(self.save_csv_dir, f"{task_id}.csv"), index=False)
            print(f"  [Success] CSV generated.")

def main():
    parser = argparse.ArgumentParser(description="ATLAS task preprocessing script")
    parser.add_argument('--task_id', type=str, help="Specific task ID to process (if omitted, scan all)")
    args = parser.parse_args()

    preprocessor = AtlasPreprocessor()

    if args.task_id:
        preprocessor.process_task(args.task_id)
    else:
        # Automatic mode: scan all sub-folders under data root
        all_tasks = [d for d in os.listdir(preprocessor.data_root)
                     if os.path.isdir(os.path.join(preprocessor.data_root, d)) and d != 'csv_files']
        for tid in all_tasks:
            preprocessor.process_task(tid)


if __name__ == "__main__":
    main()