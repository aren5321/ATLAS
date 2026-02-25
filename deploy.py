import os
import cv2
import json
import torch
import numpy as np
from task_manager import TaskManager
from assembler import AtlasAssembler


def print_logo():
    print("""
    ^        ___________  __          _        _______ 
   / \      |           ||  |        / \      /  ____/
  / ^ \     `---|  |---`|  |       / ^ \     |  |___ 
 / /_\ \        |  |    |  |      / /_\ \    `---.  \\
/  ___  \       |  |    |  |____ /  ___  \   ____|  |
|__| |__|       |__|    |_______|__| |__|  /_______/
    
    [ATLAS Framework Service - CLI Mode]
    """)


def handle_mask_storage(res, fname, tid):
    """Extract mask and save as PNG, then clean JSON data"""
    if 'mask_raw' in res['prediction']:
        mask_np = res['prediction'].pop('mask_raw')
        mask_dir = "results/masks"
        os.makedirs(mask_dir, exist_ok=True)

        # Add tid prefix to prevent overwriting between different tasks on the same image
        mask_filename = f"{fname.split('.')[0]}_{tid}_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)

        # Ensure single-channel 8-bit indexed image
        cv2.imwrite(mask_path, mask_np.astype(np.uint8))
        res['prediction']['mask_file'] = mask_filename
    return res


def run_inference_logic(engine, img, fname, tid, is_roi=False, parent_box=None):
    """General inference logic wrapper"""
    res = engine.predict(img, tid)
    res['task_id'] = tid
    if is_roi:
        res['is_roi_result'] = True
        res['parent_box'] = parent_box

    if res['type'] == 'segmentation':
        res = handle_mask_storage(res, fname, tid)
    return res


def main():
    print_logo()
    tm = TaskManager()
    tm.list_tasks()
    print("-" * 30)

    # 1. Automatically search for completed tasks
    completed_tasks = [tid for tid, cfg in tm.db.items() if cfg.get('status') == 'completed']
    if not completed_tasks:
        print("[Error] No completed tasks found. Please train models first.")
        return

    print("Available Tasks:")
    for i, tid in enumerate(completed_tasks):
        print(f" [{i}] {tid} ({tm.db[tid]['task_name']})")
    print("-" * 30)

    # 2. Interactive selection of mode and index parsing
    mode = input("\nSelect Mode: [1] Parallel [2] Cascade: ").strip()

    if mode == '1':
        idx_input = input("Select task indices (e.g. 0,2): ").strip()
        run_task_ids = [completed_tasks[int(i)] for i in idx_input.replace('，', ',').split(',')]
        det_task_id = None
        sub_task_ids = []
    elif mode == '2':
        # Improved cascade input: support formats like 0,1,2 where the first is detection and the rest are analysis
        while True:
            idx_input = input("Select indices (Detection first, e.g. 0,2): ").strip()
            try:
                indices = [int(i) for i in idx_input.replace('，', ',').split(',')]

                if len(indices) < 2:
                    print("[Error] Cascade mode requires at least 1 Detection and 1 Analysis task.")
                    continue

                # Check if the first task is actually a detection task
                first_tid = completed_tasks[indices[0]]
                first_task_type = tm.db[first_tid]['task_name'].lower()

                if first_task_type != 'detection':
                    print(f"[Error] Task '{first_tid}' is a {first_task_type.upper()} task.")
                    print("        In Cascade mode, the FIRST task MUST be 'detection'. Please re-enter.")
                    continue

                # Validation passed
                det_task_id = first_tid
                sub_task_ids = [completed_tasks[i] for i in indices[1:]]
                run_task_ids = [det_task_id] + sub_task_ids
                break

            except (ValueError, IndexError):
                print("[Error] Invalid input format or index out of range. Try again.")
    else:
        raise ValueError("No valid configuration found. Please check your input and try again.")

    # 2. Create a PDF for each sample image

    # 3. Initialize engine
    engine = AtlasAssembler(run_task_ids)
    sample_imgs = [f for f in os.listdir("sample") if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    os.makedirs("results/json", exist_ok=True)

    # 4. Inference loop
    for fname in sample_imgs:
        img_path = os.path.join("sample", fname)
        img = cv2.imread(img_path)
        if img is None: continue

        final_results = {"image_name": fname, "task_outputs": []}

        if mode == '1':
            # Parallel mode: run on whole image for each task
            for tid in run_task_ids:
                # GPU memory scheduling before inference
                engine.switch_task_on_gpu(tid)

                res = run_inference_logic(engine, img, fname, tid)
                final_results["task_outputs"].append(res)
        else:
            # Cascade mode: run detection first
            det_res = run_inference_logic(engine, img, fname, det_task_id)
            final_results["task_outputs"].append(det_res)

            box = det_res['prediction'].get('box')
            if box:
                # Enhanced cropping logic: support cascading multiple subtasks
                h, w = img.shape[:2]
                x1, y1, x2, y2 = box
                # Safe check for cropping coordinates
                roi = img[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]

                if roi.size > 0:
                    for sub_tid in sub_task_ids:
                        sub_res = run_inference_logic(engine, roi, fname, sub_tid, is_roi=True, parent_box=box)
                        final_results["task_outputs"].append(sub_res)

        # 5. Save JSON
        json_name = f"{os.path.splitext(fname)[0]}.json"
        with open(f"results/json/{json_name}", 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(f"[*] Processed {fname}")

    print("\n[Done] Results saved. Run 'visualizer.py' to view output.")


if __name__ == "__main__":
    main()
    # --- Automatically trigger visualization rendering ---
    print("\n" + "=" * 40)
    print("[*] Starting automatic visualization rendering...")
    try:
        # 1. Import AtlasVisualizer class from visualizer.py
        from visualizer import AtlasVisualizer

        # 2. Instantiate (its __init__ will automatically load TaskManager)
        viz = AtlasVisualizer()

        # 3. Call drawing method
        # Because draw_all default parameters are already results/json and results/masks, we can call it directly
        viz.draw_all()

        print("\n[OK] Automatic visualization finished! Please check result images in 'results/vis' directory.")
    except ImportError:
        print("[!] Import failed: please ensure visualizer.py exists in the same directory as deploy.py and class name is AtlasVisualizer")
    except Exception as e:
        print(f"[!] Error occurred during visualization: {e}")
    print("=" * 40 + "\n")