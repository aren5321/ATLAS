import cv2
import numpy as np
import json
import os
from task_manager import TaskManager


class AtlasVisualizer:
    def __init__(self):
        self.tm = TaskManager()
        self.task_index_map = {}
        self.next_index = 0
        self.golden_ratio = 0.618033988749895
        # Common clinical color palette (BGR)
        self.color_palette = [
            (0, 0, 0),  # 0 background (unused)
            (60, 180, 75),  # green (segmentation primary)
            (0, 90, 200),  # red (lesion)
            (200, 130, 0),  # blue-ish
            (180, 60, 180),  # purple
            (0, 140, 255),  # orange
            (180, 200, 0)  # yellow-green
        ]

    def _get_task_index(self, task_id):
        """arrange unique task_id index"""
        if task_id not in self.task_index_map:
            self.task_index_map[task_id] = self.next_index
            self.next_index += 1
        return self.task_index_map[task_id]
    def _get_color(self, index):
        """index to color"""
        hue = (index * self.golden_ratio) % 1.0
        sat = 0.65
        val = 0.85
        hsv = np.uint8([[[int(hue*179), int(sat*255), int(val*255)]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def draw_all(self, json_dir="results/json", mask_dir="results/masks", save_dir="results/vis"):
        os.makedirs(save_dir, exist_ok=True)
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        for jf in json_files:
            with open(os.path.join(json_dir, jf), 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_path = os.path.join("sample", data['image_name'])
            if not os.path.exists(img_path): continue

            canvas = cv2.imread(img_path)
            legend_info = []

            for task in data['task_outputs']:
                tid = task['task_id']
                t_type = task['type']
                pred = task['prediction']
                cfg = self.tm.get_task_config(tid)
                label_map = cfg.get('label_map', {})

                # Get cascade offset
                off_x, off_y = 0, 0
                if task.get('is_roi_result') and 'parent_box' in task:
                    off_x, off_y = task['parent_box'][0], task['parent_box'][1]

                # --- 1. Handle detection ---
                if t_type == 'detection' and 'box' in pred:
                    box = pred['box']
                    if not box: continue
                    # Detection is usually on full image, so off_x is typically 0
                    real_box = [box[0] + off_x, box[1] + off_y, box[2] + off_x, box[3] + off_y]

                    cls_id = pred.get('class_id', 1)  # default 1
                    color = self._get_color(self._get_task_index(tid))
                    cv2.rectangle(canvas, (real_box[0], real_box[1]), (real_box[2], real_box[3]), color, 3)

                    label_name = f"{tid}: {label_map.get(str(cls_id), 'Obj')}"
                    legend_info.append((label_name, color))

                # --- 2. Handle segmentation ---
                elif t_type == 'segmentation' and 'mask_file' in pred:
                    mask_path = os.path.join(mask_dir, pred['mask_file'])
                    if os.path.exists(mask_path):
                        # Read single-channel index mask
                        mask_idx = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        # Get all class IDs present in mask (excluding background 0)
                        present_classes = np.unique(mask_idx)
                        present_classes = present_classes[present_classes > 0]

                        for cls_id in present_classes:
                            color = self._get_color(self._get_task_index(tid))
                            # Extract current class mask
                            binary_mask = (mask_idx == cls_id).astype(np.uint8)

                            # Create full-image-sized empty overlay
                            full_mask_overlay = np.zeros_like(canvas)

                            # Paste ROI mask back to corresponding region in full image
                            h_m, w_m = binary_mask.shape
                            full_mask_overlay[off_y:off_y + h_m, off_x:off_x + w_m][binary_mask > 0] = color

                            # Blend into canvas
                            canvas = cv2.addWeighted(canvas, 0.85, full_mask_overlay, 0.4, 0)

                            # Append to legend list
                            class_name = label_map.get(str(cls_id), f"Class {cls_id}")
                            legend_info.append((f"{tid}: {class_name}", color))

                # --- 3. Handle regression ---
                elif t_type == 'regression' and 'keypoints' in pred:
                    pts = pred['keypoints']
                    color = self._get_color(self._get_task_index(tid))  # dedicated color for regression

                    # Restore offset and draw points
                    real_pts = []
                    for pt in pts:
                        rx, ry = int(pt[0] + off_x), int(pt[1] + off_y)
                        real_pts.append([rx, ry])
                        cv2.circle(canvas, (rx, ry), 6, color, -1)

                    if len(real_pts) >= 2:
                        canvas = self._draw_measurement(canvas, real_pts, color)

                    legend_info.append((f"{tid} Points", color))

                # --- 4. Handle classification ---
                elif t_type == 'classification' and 'class_id' in pred:
                    cls_id = pred['class_id']
                    score = pred.get('score', None)
                    color = self._get_color(self._get_task_index(tid))

                    # get cls name from label_map
                    class_name = label_map.get(str(cls_id), f"Class {cls_id}")
                    display_text = f"{tid}: {class_name}"
                    if score is not None:
                        display_text += f" ({score:.2f})"

                    legend_info.append((display_text, color))

            # --- 4. Draw side legend ---
            final_img = self._append_legend_sidebar(canvas, legend_info)

            save_path = os.path.join(save_dir, f"vis_{data['image_name']}")
            cv2.imwrite(save_path, final_img)
            print(f"[*] Visualization completed: {save_path}")

    def _draw_measurement(self, canvas, pts, color):
        for i in range(len(pts) - 1):
            cv2.line(canvas, (pts[i][0], pts[i][1]), (pts[i + 1][0], pts[i + 1][1]), color, 2)
        return canvas

    def wrap_text(self, text, max_chars=30):
        lines = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        return lines

    def _append_legend_sidebar(self, img, items):
        if not items: return img
        # Deduplicate by name
        seen = set()
        unique_items = []
        for name, color in items:
            if name not in seen:
                unique_items.append((name, color))
                seen.add(name)

        h, w = img.shape[:2]
        sidebar_w = 250
        sidebar = np.full((h, sidebar_w, 3), 45, dtype=np.uint8)  # dark background

        cv2.putText(sidebar, "ATLAS Infer", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

        for i, (name, color) in enumerate(unique_items):
            y_offset = 100 + i * 45
            cv2.rectangle(sidebar, (20, y_offset), (45, y_offset + 25), color, -1)
            # Truncate long text
            lines = self.wrap_text(name, max_chars=25)
            for j, line in enumerate(lines):
                y_line = y_offset + 18 + j * 20  # 20 px
                cv2.putText(sidebar, line, (55, y_line), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

        return np.hstack((img, sidebar))


if __name__ == "__main__":
    viz = AtlasVisualizer()
    viz.draw_all()