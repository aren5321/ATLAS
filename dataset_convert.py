"""
Convert FMC_UIA Dataset Format to ATLAS Dataset Format
"""
import os
import shutil
import pandas as pd
from glob import glob
from tqdm import tqdm


def restructure_dataset():
    # Path configuration
    csv_dir = "../data/train/csv_files"  # FMC_UIA dataset dir
    output_base = "./output"

    # 1. Aggregate all CSVs
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found, please check the path.")
        return

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Aggregation complete, total {len(combined_df)} records.")

    # 2. Process grouped by task_id
    for task_id, task_group in combined_df.groupby('task_id'):
        print(f"\nProcessing task: {task_id}")

        # Create base folders
        task_out_dir = os.path.join(output_base, str(task_id))
        img_out_dir = os.path.join(task_out_dir, "imagesTr")
        label_out_dir = os.path.join(task_out_dir, "labelsTr")

        os.makedirs(img_out_dir, exist_ok=True)

        # Track used filenames for de-duplication
        used_filenames = {}  # {original_name: count}
        new_rows = []  # store updated records (for classification/regression)

        for _, row in tqdm(task_group.iterrows(), total=len(task_group), desc="Copying files"):
            # Resolve real source image path (relative to csv_dir)
            src_img_path = os.path.normpath(os.path.join(csv_dir, row['image_path']))

            # Get original file name and extension
            orig_fname = os.path.basename(src_img_path)
            name_part, ext_part = os.path.splitext(orig_fname)

            # --- Renaming logic ---
            if orig_fname not in used_filenames:
                used_filenames[orig_fname] = 0
                final_fname = orig_fname
            else:
                used_filenames[orig_fname] += 1
                final_fname = f"{name_part}_{used_filenames[orig_fname]}{ext_part}"

            # Copy image
            dst_img_path = os.path.join(img_out_dir, final_fname)
            shutil.copy2(src_img_path, dst_img_path)

            # Build new data row
            current_row = row.to_dict()
            current_row['filename'] = final_fname  # add/modify filename column

            # --- Task-specific logic ---
            t_name = str(row['task_name']).lower()

            if t_name in ['detection', 'segmentation']:
                os.makedirs(label_out_dir, exist_ok=True)
                src_mask_path = os.path.normpath(os.path.join(csv_dir, row['mask_path']))

                # Synchronously rename label: keep same base name as image, but preserve its own suffix (.png or .xml)
                mask_ext = os.path.splitext(src_mask_path)[1]
                dst_mask_path = os.path.join(label_out_dir, os.path.splitext(final_fname)[0] + mask_ext)

                if os.path.exists(src_mask_path):
                    shutil.copy2(src_mask_path, dst_mask_path)
                else:
                    print(f"Warning: Label file not found {src_mask_path}")

            new_rows.append(current_row)

            # 3. Generate labels.csv for classification/regression tasks
            task_types = [tn.lower() for tn in task_group['task_name'].astype(str)]

            if any(tn.lower() in ['classification', 'regression'] for tn in task_types):
                labels_df = pd.DataFrame(new_rows)

                # --- Key change: simplify columns according to task type ---
                # Get effective task type (using first sample since type is consistent within task_id)
                current_task_type = task_types[0]

                if current_task_type == 'classification':
                    # Keep only filename and mask, then rename mask to label
                    keep_cols = ['filename', 'mask']
                    labels_df = labels_df[keep_cols].rename(columns={'mask': 'label'})

                elif current_task_type == 'regression':
                    # Keep filename and all columns starting with point_
                    point_cols = [col for col in labels_df.columns if col.startswith('point_')]
                    keep_cols = ['filename'] + point_cols
                    labels_df = labels_df[keep_cols]

                # Save processed CSV
                out_csv_path = os.path.join(task_out_dir, "labels.csv")
                labels_df.to_csv(out_csv_path, index=False)
                print(f"Cleaned and generated labels.csv: {out_csv_path}")



if __name__ == "__main__":
    restructure_dataset()