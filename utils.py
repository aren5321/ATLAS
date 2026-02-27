import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from task_manager import TaskManager


tm = TaskManager()
TASK_CONFIGURATIONS=tm.get_all_tasks()

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def multi_task_collate_fn(batch):
    """
    Custom collate function to handle different label shapes in multi-task learning.
    Images are stacked; labels and task_ids remain as lists.
    """
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    task_ids = [item['task_id'] for item in batch]
    
    # Stack images as they have consistent dimensions
    images = torch.stack(images, 0)
    
    return {'image': images, 'label': labels, 'task_id': task_ids}



def box_iou_ciou(b1, b2):
    """
    Compute CIoU loss.
    b1, b2: [B, 4] (x1, y1, x2, y2)
    """
    # 1. Compute intersection
    inter_min = torch.max(b1[:, :2], b2[:, :2])
    inter_max = torch.min(b1[:, 2:], b2[:, 2:])
    inter_wh = torch.clamp(inter_max - inter_min, min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    # 2. Compute union
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union_area = area1 + area2 - inter_area + 1e-7

    iou = inter_area / union_area

    # 3. Compute center distance and enclosing box
    center1 = (b1[:, :2] + b1[:, 2:]) / 2
    center2 = (b2[:, :2] + b2[:, 2:]) / 2
    rho2 = torch.sum((center1 - center2) ** 2, dim=1)

    enclose_min = torch.min(b1[:, :2], b2[:, :2])
    enclose_max = torch.max(b1[:, 2:], b2[:, 2:])
    enclose_wh = torch.clamp(enclose_max - enclose_min, min=0)
    c2 = torch.sum(enclose_wh ** 2, dim=1) + 1e-7

    # 4. Compute CIoU penalty term
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan((b1[:, 2] - b1[:, 0]) / (b1[:, 3] - b1[:, 1] + 1e-7)) -
        torch.atan((b2[:, 2] - b2[:, 0]) / (b2[:, 3] - b2[:, 1] + 1e-7)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return iou - (rho2 / c2 + alpha * v)


class ATLAS_SingleBoxLoss(nn.Module):
    """
    FCOS-lite (single GT box per image)
    outputs: [B, P, 6] = cls | ctr | l t r b
    labels : [B, 4] xyxy normalized
    """
    def __init__(
        self,
        input_size=(512, 512),
        strides=(8, 16, 32),
        center_radius=1.5,
        lambda_cls=1.0,
        lambda_reg=1.0,
        lambda_ctr=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.strides = strides
        self.center_radius = center_radius
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_ctr = lambda_ctr

    # --------------------------------------------------------
    # target builder (unchanged)
    # --------------------------------------------------------
    @torch.no_grad()
    def build_targets(self, labels, device):
        B = labels.shape[0]
        H, W = self.input_size

        points_all, stride_all = [], []
        for s in self.strides:
            fh, fw = H // s, W // s
            ys, xs = torch.meshgrid(
                torch.arange(fh, device=device),
                torch.arange(fw, device=device),
                indexing="ij",
            )
            pts = torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], dim=-1)
            points_all.append(pts.reshape(-1, 2))
            stride_all.append(torch.full((fh * fw,), s, device=device))

        points = torch.cat(points_all, dim=0)
        stride_map = torch.cat(stride_all, dim=0)

        P = points.shape[0]
        cls_t = torch.zeros((B, P), device=device)
        ctr_t = torch.zeros((B, P), device=device)
        reg_t = torch.zeros((B, P, 4), device=device)
        pos_m = torch.zeros((B, P), dtype=torch.bool, device=device)

        for b in range(B):
            x1 = labels[b, 0] * W
            y1 = labels[b, 1] * H
            x2 = labels[b, 2] * W
            y2 = labels[b, 3] * H

            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            px, py = points[:, 0], points[:, 1]
            l = px - x1
            t = py - y1
            r = x2 - px
            btm = y2 - py

            inside = (l > 0) & (t > 0) & (r > 0) & (btm > 0)
            radius = self.center_radius * stride_map
            center = (torch.abs(px - cx) < radius) & (torch.abs(py - cy) < radius)
            pos = inside & center
            if pos.sum() == 0:
                continue

            cls_t[b, pos] = 1.0

            l_pos, t_pos, r_pos, b_pos = l[pos], t[pos], r[pos], btm[pos]
            ctr = torch.sqrt(
                (torch.min(l_pos, r_pos) / torch.max(l_pos, r_pos)) *
                (torch.min(t_pos, b_pos) / torch.max(t_pos, b_pos))
            )
            ctr_t[b, pos] = ctr

            reg = torch.stack([l_pos, t_pos, r_pos, b_pos], dim=-1)
            reg_t[b, pos] = reg / stride_map[pos].unsqueeze(-1)
            pos_m[b, pos] = True

        return cls_t, ctr_t, reg_t, pos_m, points, stride_map

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(self, outputs, labels):
        B, P, _ = outputs.shape
        device = outputs.device

        cls_p = outputs[..., 0].sigmoid()
        ctr_p = outputs[..., 1].sigmoid()
        reg_p = torch.exp(outputs[..., 2:6])

        cls_t, ctr_t, reg_t, pos_m, points, stride_map = \
            self.build_targets(labels, device)

        num_pos = pos_m.sum().clamp(min=1)

        # ---------------- cls ----------------
        cls_loss = F.binary_cross_entropy(cls_p, cls_t, reduction="mean")

        # ---------------- ctr ----------------
        ctr_loss = F.binary_cross_entropy(
            ctr_p[pos_m], ctr_t[pos_m], reduction="sum"
        ) / num_pos

        # ---------------- reg (log-IoU) ----------------
        px = points[:, 0].unsqueeze(0)
        py = points[:, 1].unsqueeze(0)

        reg_p_abs = reg_p * stride_map.view(1, -1, 1)
        reg_t_abs = reg_t * stride_map.view(1, -1, 1)

        x1p = px - reg_p_abs[..., 0]
        y1p = py - reg_p_abs[..., 1]
        x2p = px + reg_p_abs[..., 2]
        y2p = py + reg_p_abs[..., 3]

        x1t = px - reg_t_abs[..., 0]
        y1t = py - reg_t_abs[..., 1]
        x2t = px + reg_t_abs[..., 2]
        y2t = py + reg_t_abs[..., 3]

        box_p = torch.stack([x1p, y1p, x2p, y2p], dim=-1)[pos_m]
        box_t = torch.stack([x1t, y1t, x2t, y2t], dim=-1)[pos_m]

        # IoU
        inter_w = torch.clamp(
            torch.min(box_p[:, 2], box_t[:, 2]) -
            torch.max(box_p[:, 0], box_t[:, 0]), min=0
        )
        inter_h = torch.clamp(
            torch.min(box_p[:, 3], box_t[:, 3]) -
            torch.max(box_p[:, 1], box_t[:, 1]), min=0
        )
        inter = inter_w * inter_h

        area_p = (box_p[:, 2] - box_p[:, 0]) * (box_p[:, 3] - box_p[:, 1])
        area_t = (box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1])
        union = area_p + area_t - inter + 1e-7
        iou = inter / union

        # center-ness weight
        ctr_w = ctr_t[pos_m]

        # scale weight (only affects gradients)
        w = torch.ones_like(iou)
        w[area_t < (24 ** 2)] = 2.0

        w[area_t > (96 ** 2)] = 0.7

        reg_loss = (w * (-torch.log(iou + 1e-6)) * ctr_w).sum() / num_pos

        total = (
            self.lambda_cls * cls_loss +
            self.lambda_ctr * ctr_loss +
            self.lambda_reg * reg_loss
        )
        return total


@torch.no_grad()
def get_pred_boxes(
    outputs,
    input_size=(512, 512),
    strides=(8, 16, 32),
    score_thr=0.05,
    topk=10,   # per scale
):
    device = outputs.device
    B, P, _ = outputs.shape
    H, W = input_size

    cls_p = outputs[..., 0].sigmoid()
    ctr_p = outputs[..., 1].sigmoid()
    reg_p = torch.exp(outputs[..., 2:6])

    scores = cls_p * ctr_p

    # ---------------- points & stride ----------------
    points, stride_map, scale_ids = [], [], []
    sid = 0
    for s in strides:
        fh, fw = H // s, W // s
        ys, xs = torch.meshgrid(
            torch.arange(fh, device=device),
            torch.arange(fw, device=device),
            indexing="ij",
        )
        pts = torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], dim=-1)
        n = fh * fw
        points.append(pts.reshape(-1, 2))
        stride_map.append(torch.full((n,), s, device=device))
        scale_ids.append(torch.full((n,), sid, device=device))
        sid += 1

    points = torch.cat(points, dim=0)
    stride_map = torch.cat(stride_map, dim=0)
    scale_ids = torch.cat(scale_ids, dim=0)

    all_boxes = []

    for b in range(B):
        boxes_all, scores_all = [], []

        for sid in range(len(strides)):
            mask = scale_ids == sid
            sc = scores[b][mask]
            if sc.numel() == 0:
                continue

            keep = sc > score_thr
            if keep.sum() == 0:
                continue

            sc = sc[keep]
            reg = reg_p[b][mask][keep]
            pts = points[mask][keep]
            stride = stride_map[mask][keep]

            k = min(topk, sc.numel())
            sc, idx = torch.topk(sc, k)

            reg = reg[idx]
            pts = pts[idx]
            stride = stride[idx]

            reg_abs = reg * stride.unsqueeze(-1)

            x1 = (pts[:, 0] - reg_abs[:, 0]).clamp(0, W)
            y1 = (pts[:, 1] - reg_abs[:, 1]).clamp(0, H)
            x2 = (pts[:, 0] + reg_abs[:, 2]).clamp(0, W)
            y2 = (pts[:, 1] + reg_abs[:, 3]).clamp(0, H)

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            boxes_all.append(boxes)
            scores_all.append(sc)

        if len(boxes_all) == 0:
            all_boxes.append(torch.tensor([0, 0, 0, 0], device=device))
            continue

        boxes = torch.cat(boxes_all, dim=0)
        scores_f = torch.cat(scores_all, dim=0)

        best = scores_f.argmax()
        box = boxes[best]

        all_boxes.append(
            torch.tensor(
                [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                device=device,
            )
        )

    return torch.stack(all_boxes, dim=0)


#------------------ Detection loss related --------------------------

# --- Metric Calculations ---

def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_dice_coefficient(y_true, y_pred_logits):
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = y_pred_logits.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_one_hot = F.one_hot(y_pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

def calculate_mae(y_true, y_pred, image_size=(256, 256)):
    h, w = image_size
    y_true_px = y_true.cpu().numpy().copy()
    y_pred_px = y_pred.cpu().numpy().copy()
    y_true_px[:, 0::2] *= w; y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w; y_pred_px[:, 1::2] *= h
    return np.mean(np.abs(y_true_px - y_pred_px))

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy(); y_pred = y_pred.cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true, box_pred = y_true[i], y_pred[i]
        xA = max(box_true[0], box_pred[0]); yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2]); yB = min(box_true[3], box_pred[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
        box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(iou)
    return np.mean(batch_ious)

def evaluate(model, val_loader, device):
    """
    Evaluation loop supporting multi-task batches.
    """
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']

            unique_tasks_in_batch = set(task_ids)

            for task_id in unique_tasks_in_batch:
                task_indices = [i for i, t_id in enumerate(task_ids) if t_id == task_id]
                task_images = images[task_indices]
                
                # Extract and stack labels for the current task
                task_labels_list = [labels[i] for i in task_indices]
                task_labels = torch.stack(task_labels_list, 0)
                
                outputs = model(task_images, task_id=task_id)
                task_name = task_id_to_name[task_id]
                
                if task_name == 'classification':
                    task_metrics[task_id]['Accuracy'].append(calculate_accuracy(task_labels, outputs))
                    task_metrics[task_id]['F1-Score'].append(calculate_f1_score(task_labels, outputs))
                
                elif task_name == 'segmentation':
                    task_metrics[task_id]['Dice'].append(calculate_dice_coefficient(task_labels.to(device), outputs))
                
                elif task_name == 'regression':
                    task_metrics[task_id]['1/MAE (pixels)'].append(1/calculate_mae(task_labels, outputs))
                
                elif task_name == 'detection':
                    # 1. Dynamically get current task image H and W (shape [B, C, H, W])
                    curr_h, curr_w = task_images.shape[2], task_images.shape[3]

                    # 2. Pass size into get_pred_boxes
                    pred_boxes = get_pred_boxes(outputs, input_size=(curr_h, curr_w))
                    # pred_boxes = get_pred_boxes(outputs)

                    iou = calculate_iou(task_labels, pred_boxes)
                    task_metrics[task_id]['IoU'].append(iou)

    results = []
    sorted_task_ids = sorted(list(task_id_to_name.keys()))
    for task_id in sorted_task_ids:
        if task_id in task_metrics:
            task_name = task_id_to_name[task_id]
            result_row = {'Task ID': task_id, 'Task Name': task_name}
            for metric_name, values in task_metrics[task_id].items():
                result_row[metric_name] = np.mean(values)
            results.append(result_row)
    return pd.DataFrame(results)