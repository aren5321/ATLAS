import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from task_manager import TaskManager
import cv2

class AtlasVisualizerPDFFixed:
    def __init__(self):
        self.tm = TaskManager()
        self.task_index_map = {}
        self.next_index = 0
        self.golden_ratio = 0.618033988749895

    def _get_task_index(self, task_id):
        if task_id not in self.task_index_map:
            self.task_index_map[task_id] = self.next_index
            self.next_index += 1
        return self.task_index_map[task_id]

    def _get_color(self, index):
        import colorsys
        hue = (index * self.golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
        return (r, g, b)

    def draw_all_pdf(self, json_dir="results/json", mask_dir="results/masks", gt_dir="results/gt",
                     save_dir="results/pdf", ncol=2):
        os.makedirs(save_dir, exist_ok=True)
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        for jf in json_files:
            with open(os.path.join(json_dir, jf), 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_path = os.path.join("sample", data['image_name'])
            if not os.path.exists(img_path): continue

            canvas = cv2.imread(img_path)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w = canvas.shape[:2]

            # fig_h = h/100 + 1
            # fig_w = w/100
            # fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

            ax.imshow(canvas)
            ax.set_xlim([0, w])
            ax.set_ylim([h, 0])
            ax.axis('off')

            legend_info = []  # (text, color or None)

            for task in data['task_outputs']:
                tid = task['task_id']
                t_type = task['type']
                pred = task['prediction']
                cfg = self.tm.get_task_config(tid)
                label_map = cfg.get('label_map', {})

                off_x, off_y = 0, 0
                if task.get('is_roi_result') and 'parent_box' in task:
                    off_x, off_y = task['parent_box'][0], task['parent_box'][1]

                color = self._get_color(self._get_task_index(tid))

                # --- Detection ---
                if t_type=='detection' and 'box' in pred and pred['box']:
                    x1, y1, x2, y2 = [v+off for v, off in zip(pred['box'], [off_x, off_y, off_x, off_y])]
                    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, edgecolor=color, facecolor='none', lw=2))
                    legend_info.append((f"{tid}: {label_map.get(str(pred.get('class_id',1)),'Obj')}", color))

                # --- Segmentation ---
                elif t_type=='segmentation' and 'mask_file' in pred:
                    # pred mask
                    mask_path = os.path.join(mask_dir, pred['mask_file'])
                    if os.path.exists(mask_path):
                        mask_idx = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        classes = np.unique(mask_idx)
                        classes = classes[classes>0]
                        for cls_id in classes:
                            ys, xs = np.where(mask_idx==cls_id)
                            if len(xs)==0: continue
                            poly_coords = np.column_stack((xs+off_x, ys+off_y))
                            ax.add_patch(Polygon(poly_coords, closed=False, facecolor=color, edgecolor=color, alpha=0.7))
                            legend_info.append((f"{tid}: {label_map.get(str(cls_id), f'Class {cls_id}')}", color))

                    # GT mask contour
                    gt_mask_path = os.path.join(gt_dir, pred['mask_file'])
                    if os.path.exists(gt_mask_path):
                        gt_idx = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                        #  get contours
                        contours, _ = cv2.findContours(gt_idx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            cnt = cnt.squeeze()
                            if cnt.ndim == 1:
                                continue
                            ax.add_patch(Polygon(cnt, closed=True, facecolor='none', edgecolor='red', lw=2))
                        legend_info.append((f"{tid} GT", 'red'))


                # --- Regression ---
                elif t_type=='regression' and 'keypoints' in pred:
                    pts = np.array(pred['keypoints'])
                    pts[:,0] += off_x
                    pts[:,1] += off_y
                    ax.scatter(pts[:,0], pts[:,1], s=30, c=[color], edgecolors='black')
                    legend_info.append((f"{tid} Points", color))

                # --- Classification ---
                elif t_type=='classification' and 'class_id' in pred:
                    cls_id = pred['class_id']
                    score = pred.get('score', None)
                    text = f"{tid}: {label_map.get(str(cls_id), f'Class {cls_id}')}"
                    if score is not None: text+=f" ({score:.2f})"
                    legend_info.append((text, None))  # no color for cls task

            # --- bottem legend ---
            self._draw_bottom_legend(ax, legend_info, ncol=ncol, img_w=w, img_h=h)

            # --- save PDF ---
            from matplotlib.backends.backend_pdf import PdfPages
            save_path = os.path.join(save_dir, f"{os.path.splitext(data['image_name'])[0]}.pdf")
            pp = PdfPages(save_path)
            pp.savefig(fig, bbox_inches='tight', dpi=300)
            pp.close()
            plt.close(fig)
            print(f"[*] Saved PDF: {save_path}")

    def _draw_bottom_legend(self, ax, items, ncol=1, img_w=512, img_h=512):
        if not items: return

        # 去重
        seen = set()
        unique_items = []
        for name, color in items:
            if name not in seen:
                unique_items.append((name, color))
                seen.add(name)

        handles = []
        labels = []
        for text, color in unique_items:
            if color is None:
                handles.append(None)
            else:
                rect = Rectangle((0,0),1,1, facecolor=color, edgecolor='black')
                handles.append(rect)
            labels.append(text)

        # filter None handle
        handles_filtered = [h if h is not None else Rectangle((0,0),0,0, facecolor='none', edgecolor='none') for h in handles]

        ax.legend(handles_filtered, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fontsize=12, ncol=ncol, frameon=False, handlelength=1.5, handleheight=1.5)

if __name__=="__main__":
    viz = AtlasVisualizerPDFFixed()
    viz.draw_all_pdf(ncol=2)
