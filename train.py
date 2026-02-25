import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import os
import logging
import json
from task_manager import TaskManager
import argparse
# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory
from utils import (
    multi_task_collate_fn,
    evaluate,
    ATLAS_SingleBoxLoss,
    set_seed
)
# 1. Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, required=True)
parser.add_argument('--resume_from', type=str, default=None, help='Warm start from weights of which task ID')
parser.add_argument('--mtl_weights', type=str, default=None, help='shared backbone MTL weights dir')
args = parser.parse_args()

# 2. Automatically get config
tm = TaskManager()
cfg = tm.get_task_config(args.task_id)


# --- After modification ---
# Directly use real-time config from tm to construct a list that matches Factory requirements
current_task_config = {
    'task_id': args.task_id,
    'task_name': cfg['task_name'],
    'num_classes': cfg['num_classes'],
    # If Factory needs more fields, pull from cfg
}
TASK_CONFIGURATIONS = [current_task_config]


# Training configuration
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 100
DATA_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42
MODEL_SAVE_PATH = '.pth'
VAL_SPLIT = 0.1
TES_SPLIT = 0.2
PATIENCE = 20
WEIGHT_PATH='weights'

TASK_ID = args.task_id


def _is_numeric(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def main():
    history = {
        'train_loss': defaultdict(list),  # training loss for each task
        'val_score': [],  # validation average score for each epoch
        'tes_score': []  # test average score for each epoch
    }

    # --- Setup logging ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(WEIGHT_PATH, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, f"{TASK_ID}_train.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    # --------------------------------------------

    set_seed(RANDOM_SEED)

    # Define worker seed init function
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create DataLoader generator with fixed seed
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device used: {device}")
    model = MultiTaskModelFactory(task_configs=TASK_CONFIGURATIONS).to(device)
    # ---  save initial weights for TIES-Merging ---
    # initial_template_path = "initial_template.pth"
    # if not os.path.exists(initial_template_path):
    #     torch.save(model.state_dict(), initial_template_path)
    #     print(f"Initial template (untrained weights) saved to {initial_template_path}")
    # ------------------------------

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {params / 1000:.2f} K")
    model_name = args.task_id

    # --- Warm start logic ---
    if args.mtl_weights:
        #  Factory load shared backbone function
        success = model.load_shared_multitask_weights(args.mtl_weights, strict=False)
        if success:
            logging.info(f"[*] success load shared MTL backbone weight: {args.mtl_weights}")
            args.resume_from = "MTL_SHARED"  # triger resume_from
        else:
            logging.error(f"[!] load failed!")
    elif args.resume_from:
        # Assume weights are in weights folder and named as task_id.pth
        resume_path = os.path.join(WEIGHT_PATH, f"{args.resume_from}.pth")
        # strict=False is key for transfer learning: allows same backbone but different heads (e.g. class count)
        model.load_task_weights(args.task_id, resume_path, strict=False)
        logging.info(f"[*] Transfer learning enabled: loading pretrained features from {args.resume_from}")
    # --------------------

    params = sum(p.numel() for p in model.parameters())

    # After extracting target size, force to multiples of 32
    target_h = (cfg.get('resample_target', [512, 512])[0] // 32) * 32
    target_w = (cfg.get('resample_target', [512, 512])[1] // 32) * 32

    logging.info(f"Target Resize: {target_h}x{target_w} (from tasks_db.json)")

    # Data loading and splitting
    # Training transforms with augmentation
    # Common params for transforms
    BBOX_PARAMS = A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        clip=True,
        min_visibility=0.1
    )

    KEYPOINT_PARAMS = A.KeypointParams(
        format='xy',
        remove_invisible=False  # keep coordinates even if points are cropped to avoid dimension collapse
    )

    train_transforms = A.Compose([
        A.Resize(target_h, target_w),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=BBOX_PARAMS, keypoint_params=KEYPOINT_PARAMS,seed=RANDOM_SEED)

    val_transforms = A.Compose([
        A.Resize(target_h, target_w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=BBOX_PARAMS, keypoint_params=KEYPOINT_PARAMS,seed=RANDOM_SEED)

    # Create full dataset to get indices
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms, task_id=TASK_ID)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    tes_size=int(dataset_size * TES_SPLIT)
    train_size = dataset_size - val_size-tes_size

    # Split indices
    # Get all original indices (based on temp dataset)
    full_indices = list(range(dataset_size))
    rng = np.random.default_rng(RANDOM_SEED)  # use independent generator so as not to pollute global random
    rng.shuffle(full_indices)


    #  Manually split
    train_end = train_size
    val_end = train_size + val_size

    tr_indices = full_indices[:train_end]
    val_indices = full_indices[train_end:val_end]
    te_indices = full_indices[val_end:]

    # Pass indices directly when constructing; Dataset will do iloc and reset_index internally.
    # This yields Dataset objects instead of Subset, preserving attributes (e.g., dataframe).
    train_dataset = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        transforms=train_transforms,
        task_id=TASK_ID,
        indices=tr_indices  # pass directly
    )

    val_dataset = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        transforms=val_transforms,
        task_id=TASK_ID,
        indices=val_indices
    )

    tes_dataset = MultiTaskDataset(
        data_root=DATA_ROOT_PATH,
        transforms=val_transforms,
        task_id=TASK_ID,
        indices=te_indices
    )

    # 4. DataLoaders become very clean
    train_sampler = MultiTaskUniformSampler(train_dataset, batch_size=BATCH_SIZE, seed=RANDOM_SEED)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
        worker_init_fn=seed_worker,
        generator=dataloader_generator
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    tes_loader = torch.utils.data.DataLoader(
        tes_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )

    # Model and loss setup
    # If TASK_ID is specified, only create config for that task_id
    if TASK_ID is not None:
        task_configs = [cfg for cfg in TASK_CONFIGURATIONS if cfg['task_id'] == TASK_ID]
        if not task_configs:
            raise ValueError(f"Specified TASK_ID '{TASK_ID}' does not exist in TASK_CONFIGURATIONS")
        logging.info(f"Training only task_id: {TASK_ID}")
    else:
        task_configs = TASK_CONFIGURATIONS
        logging.info(f"Training all task_ids")

    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'),
        'classification': nn.CrossEntropyLoss(),
        'regression': nn.MSELoss(),
        'detection': ATLAS_SingleBoxLoss(input_size=(target_h, target_w),)
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
    # Print available task_ids
    logging.info(f" Available Task IDs (from TASK_CONFIGURATIONS): {list(task_id_to_name.keys())}")

    # Optimization setup
    logging.info("\n--- Setting parameter groups with differential LRs ---")
    param_groups = []

    # Backbone/Neck lr down if warm start
    backbone_lr_factor = 0.1 if args.resume_from else 1.0

    for task_id, sub_model in model.models.items():
        backbone_params = []
        head_params = []

        for name, param in sub_model.named_parameters():
            if not param.requires_grad:
                continue
            # if head: LR invariant ，else:（backbone/neck）down
            if 'head' in name.lower():
                head_params.append(param)
            else:
                backbone_params.append(param)

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': LEARNING_RATE * backbone_lr_factor,
                'name': f"{task_id}_backbone"
            })
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': LEARNING_RATE,
                'name': f"{task_id}_head"
            })

        logging.info(f"  - Task '{task_id}':")
        logging.info(f"    - Backbone/Neck LR: {LEARNING_RATE * backbone_lr_factor:.6f}")
        logging.info(f"    - Head LR:          {LEARNING_RATE:.6f}")

    optimizer = optim.AdamW(param_groups)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    logging.info("\n--- Cosine Annealing Scheduler configured ---")

    best_val_score = -float('inf')
    early_stop_counter = 0  # early-stop counter: number of consecutive epochs without improvement
    logging.info("\n" + "=" * 50 + "\n--- Start Training ---")
    model_save_path= os.path.join(WEIGHT_PATH,f"{TASK_ID}{MODEL_SAVE_PATH}")


    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            outputs = model(images, task_id=current_task_id)

            loss = loss_functions[task_name](outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Train reporting
        logging.info("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            history['train_loss'][task_id].append(float(avg_loss))
            logging.info(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        logging.info("-" * 40)

        # Validation
        val_results_df = evaluate(model, val_loader, device)

        score_cols = [col for col in val_results_df.columns if
                      'xxx' not in col and _is_numeric(val_results_df[col].iloc[0])]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        logging.info("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            logging.info(val_results_df.to_string(index=False))
        logging.info(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")
        history['val_score'].append(float(avg_val_score))

        print("Save train_history.json........")
        train_history_path = os.path.join(log_dir, f"{TASK_ID}_{model_name}_train_history.json")
        with open(train_history_path , "w") as f:
            # Convert dict -> list
            json.dump({
                'train_loss': {str(k): v for k, v in history['train_loss'].items()},
                'val_score': history['val_score']
            }, f)

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")
            early_stop_counter = 0  # reset when validation improves
        else:
            early_stop_counter += 1  # no improvement, increase counter
            logging.info(f"-> No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")

            # Trigger early stop if needed
            if early_stop_counter >= PATIENCE:
                logging.info(f"\nEarly stopping triggered! No improvement for {PATIENCE} consecutive epochs.")
                logging.info(f"Best validation score: {best_val_score:.4f}")
                break  # stop training loop

        # Update scheduler
        scheduler.step()

    logging.info(f"\n--- Training Finished ---\nBest model saved at: {model_save_path}")

    # Test Duration
    tes_results_df = evaluate(model, tes_loader, device)

    # 2. Extract metrics dict for current task
    # Find row whose Task ID equals current task id
    current_task_row = tes_results_df[tes_results_df['Task ID'] == args.task_id]

    if not current_task_row.empty:
        # Convert to dict and remove 'Task ID', 'Task Name', and NaN values
        best_scores = current_task_row.iloc[0].to_dict()
        # Use v == v trick (NaN != NaN) to filter NaNs
        best_scores = {k: v for k, v in best_scores.items()
                       if k not in ['Task ID', 'Task Name'] and v == v}

        # 3. Write back to task manager
        tm.update_training_result(args.task_id, best_scores)
        logging.info(f"Updated Task {args.task_id} results with: {best_scores}")

    score_cols = [col for col in tes_results_df.columns if
                  'xxx' not in col and _is_numeric(tes_results_df[col].iloc[0])]
    avg_tes_score = 0
    if not tes_results_df.empty and score_cols:
        avg_tes_score = tes_results_df[score_cols].mean().mean()

    logging.info("\n---  Test Report ---")
    if not tes_results_df.empty:
        logging.info(tes_results_df.to_string(index=False))
    logging.info(f"--- Average Test Score (Higher is better): {avg_tes_score:.4f} ---")
    history['tes_score'].append(float(avg_tes_score))


    print("Save Test_history.json........")
    test_history_path = os.path.join(log_dir, f"{TASK_ID}_{model_name}_test_history.json")
    with open(test_history_path, "w") as f:
        # Convert dict -> list
        json.dump({
            'val_score': history['val_score']
        }, f)


if __name__ == '__main__':
    main()