import torch
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import local modules
from task_manager import TaskManager
from model_factory import MultiTaskModelFactory
from utils import get_pred_boxes  # reuse detection decoding logic from utils


class AtlasAssembler:
    def __init__(self, task_id_list, device=None):
        """
        Initialize model assembler.
        :param task_id_list: list of task IDs to load, e.g. ['thyroid_nodule_det', 'breast_2cls']
        :param device: 'cuda' or 'cpu'; if None, auto-detect
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Assembler init (Device: {self.device}) ---")

        self.tm = TaskManager()

        # 1. Prepare config list
        configs = []
        for tid in task_id_list:
            try:
                # Get config, including num_classes, task_name, resample_target, etc.
                cfg = self.tm.get_task_config(tid)
                # Ensure task_id field exists (required by Factory)
                cfg['task_id'] = tid
                configs.append(cfg)
            except ValueError:
                print(f"  [Error] Task {tid} not found in tasks_db.json, skipped.")

        if not configs:
            raise ValueError("No valid task configs found; initialization failed.")

        # 2. Instantiate multi-task model (Factory automatically creates multiple heads)
        # See model_factory.py for details
        self.model = MultiTaskModelFactory(configs).to(self.device)
        self.model.eval()

        # 3. Load weights & build dedicated preprocessors
        self.transforms = {}
        self.configs_cache = {}  # cache configs for inference (e.g. num_classes)

        for cfg in configs:
            tid = cfg['task_id']
            self.configs_cache[tid] = cfg

            #  Load weights
            # train.py saves to weights/{task_id}.pth
            weight_path = cfg.get('weight_path', f"weights/{tid}.pth")

            # Call powerful load_task_weights method from model_factory.py
            # It automatically strips 'models.task_id.' prefix to match train.py format
            if hasattr(self.model, 'load_task_weights'):
                self.model.load_task_weights(tid, weight_path, strict=False)
            else:
                # Defensive code: if factory version is incompatible
                print(f"  [Error] Factory is missing load_task_weights method, cannot load weights")

            #  Build preprocessing pipeline (automatically align with training size)
            # Get resample_target from DB (format [H, W]); default to 512 if not set.
            # Note: Albumentations Resize takes (height, width)
            target_shape = cfg.get('resample_target', [512, 512])
            h, w = target_shape[0], target_shape[1]

            # Must keep normalization parameters consistent with train.py
            self.transforms[tid] = A.Compose([
                A.Resize(h, w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            print(f"  [Ready] Task {tid} (input size: {h}x{w})")

    @torch.no_grad()
    def predict(self, image_input, task_id):
        """
        Unified inference interface.
        :param image_input: image path (str) or OpenCV image array (np.ndarray, BGR format)
        :param task_id: which task to run
        :return: formatted JSON dict
        """
        if task_id not in self.transforms:
            return {"error": f"Task {task_id} not loaded in Assembler"}

        # 1. Read image and store original size
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {"error": f"Image path not found: {image_input}"}
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                return {"error": "Failed to decode image"}
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            # Assume input is BGR image read by OpenCV
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            return {"error": "Unsupported image format"}

        h_ori, w_ori = img_rgb.shape[:2]

        # 2. Preprocess
        transform = self.transforms[task_id]
        augmented = transform(image=img_rgb)
        # [3, H, W] -> [1, 3, H, W]
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # 3. Model inference
        # MultiTaskModelFactory forward needs task_id to route to correct head
        raw_output = self.model(img_tensor, task_id=task_id)

        # 4. Post-process and restore coordinates
        # Pass img_tensor.shape[2:] i.e. (H_model, W_model) for decoding detection boxes
        cfg = self.configs_cache[task_id]
        result = self._post_process(raw_output, cfg, (h_ori, w_ori), img_tensor.shape[2:])

        return result

    def _post_process(self, output, cfg, ori_shape, model_input_shape):
        """
        Core: parse model output into clinically readable format and restore coordinates.
        """
        task_name = cfg['task_name'].lower()
        label_map = cfg.get('label_map', {})
        h_ori, w_ori = ori_shape
        h_model, w_model = model_input_shape

        res = {"task_id": cfg['task_id'], "type": task_name}

        # --- Classification task ---
        if task_name == 'classification':
            # output: [1, num_classes]
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])

            res['prediction'] = {
                "class_id": int(pred_idx),
                "label": label_map.get(str(pred_idx), f"Class {pred_idx}"),
                "confidence": round(confidence, 4),
                "probs": {label_map.get(str(i), str(i)): round(float(p), 4) for i, p in enumerate(probs)}
            }

        # --- Segmentation task ---
        elif task_name == 'segmentation':
            # output: [1, num_classes, H, W] or [1, 1, H, W]
            if output.shape[1] == 1:
                # Binary segmentation, use Sigmoid
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                # Multi-class segmentation, use Softmax + Argmax
                mask = torch.softmax(output, dim=1).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Resize back to original image size (nearest neighbor to keep pixel as int)
            mask_resized = cv2.resize(mask, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)

            # Compute lesion area ratio (example metric)
            area_ratio = np.sum(mask_resized > 0) / (h_ori * w_ori) if (h_ori * w_ori) > 0 else 0

            res['prediction'] = {
                "mask_raw": mask_resized,  # mask contains pixel values 0, 1, 2, ...
                "mask_shape": mask_resized.shape,
                "lesion_area_ratio": round(float(np.sum(mask_resized > 0) / (h_ori * w_ori)), 4)
            }
            # Deployment note: mask_resized can be encoded as Base64 string and returned to frontend

        # --- Keypoint regression ---
        elif task_name == 'regression':
            # output: [1, num_points * 2]
            # Normalization is done in dataset.py, so outputs here are in [0, 1]
            coords = output.cpu().numpy()[0]
            points = []
            for i in range(0, len(coords), 2):
                x_norm, y_norm = coords[i], coords[i + 1]
                # Restore coordinates
                x_pixel = int(x_norm * w_ori)
                y_pixel = int(y_norm * h_ori)
                points.append([x_pixel, y_pixel])

            res['prediction'] = {"keypoints": points}

        # --- Detection task ---
        elif task_name == 'detection':
            # 1. Use decoding function in utils.py
            # CRITICAL: must pass current tensor (H, W) otherwise strides are wrong and boxes shift
            norm_boxes = get_pred_boxes(output, input_size=(h_model, w_model))  # returns [B, 4]

            # utils.py returns normalized coords [x1/W, y1/H, x2/W, y2/H]
            if norm_boxes.shape[0] > 0:
                box = norm_boxes[0].cpu().numpy()  # take first box (single-target)

                # 2. Map back to original image
                x1 = int(box[0] * w_ori)
                y1 = int(box[1] * h_ori)
                x2 = int(box[2] * w_ori)
                y2 = int(box[3] * h_ori)

                res['prediction'] = {
                    "box": [x1, y1, x2, y2],
                    "label": "Target"  # ATLAScore single-class detection usually has one foreground class
                }
            else:
                res['prediction'] = {"box": [], "note": "No object detected"}

        return res

    # ensure only current task on GPU, others on CPU
    def switch_task_on_gpu(self, target_task_id):
        """ensure only current task on GPU, others on CPU"""
        for tid, model_branch in self.model.models.items():
            if tid == target_task_id:
                model_branch.to(self.device)  # to GPU
            else:
                model_branch.to('cpu')  # to CPU release GPU memory

        # force GPU memory to be released
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================================================
# Quick test entrypoint
# =========================================================
if __name__ == "__main__":
    # Assume train.py has been run and tasks_db.json and weights/ exist
    # Replace with actual task IDs
    test_tasks = ['thyroid_nodule_det','organ_cls']

    try:
        # 1. Initialize engine
        assembler = AtlasAssembler(test_tasks)

        # 2. Build a dummy image (all zeros) to test pipeline
        print("\n>>> Generating test image...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # 3. Run inference
        print(f"\n>>> Running inference test for {test_tasks[0]}:")
        res_det = assembler.predict(dummy_img, test_tasks[0])
        print("Inference result:", res_det)

        print(f"\n>>> Running inference test for {test_tasks[1]}:")
        res_det = assembler.predict(dummy_img, test_tasks[1])
        print("Inference result:", res_det)

        print("\n[Pass] Assembler self-check passed!")

    except Exception as e:
        print(f"\n[Error] Initialization or inference failed: {e}")
        print("Hint: ensure tasks_db.json exists and corresponding .pth files are in weights/ directory")