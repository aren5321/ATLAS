import torch
import torch.nn as nn
import os
from typing import List, Dict
from modelzoom.ATLAScore import ATLAScore


class MultiTaskModelFactory(nn.Module):
    def __init__(self, task_configs: List[Dict]):
        """
        task_configs: Format example [{'task_id': 'FUGC', 'task_name': 'regression', 'num_classes': 2}, ...]
        """
        super().__init__()
        self.models = nn.ModuleDict()

        # Unified task type mapping (corresponding to ATLAS-core)
        self.name_to_id = {
            'classification': 0,
            'segmentation': 1,
            'regression': 2,
            'detection': 3
        }

        print(f"--- ModelFactory: Initializing modules ---")
        for config in task_configs:
            tid = config['task_id']
            t_name = config['task_name'].lower()
            num_cls = config['num_classes']

            # Get integer ID required internally by ATLAScore
            atlas_task_id = self.name_to_id.get(t_name)
            if atlas_task_id is None:
                print(f"  [Error] Unknown task type: {t_name}")
                continue

            # Special handling for regression tasks: coordinate points are usually num_classes * 2
            actual_classes = num_cls * 2 if t_name == 'regression' else num_cls

            # Instantiation
            self.models[tid] = ATLAScore(
                in_ch=3,
                num_classes=actual_classes,
                task_id=atlas_task_id
            )
            print(f"  [+] Successfully created branch: {tid} Type: {t_name}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        if task_id not in self.models:
            raise ValueError(f"Task ID '{task_id}' is not instantiated in the Factory.")
        return self.models[task_id](x)

    def load_task_weights(self, task_id: str, weight_path: str, strict: bool = False):
        if task_id not in self.models:
            return

        if not os.path.exists(weight_path):
            print(f"  [Warning] Weight file does not exist: {weight_path}")
            return

        state_dict = torch.load(weight_path, map_location='cpu')
        model_state_dict = self.models[task_id].state_dict()  # Get parameter structure of current model

        new_sd = {}
        for k, v in state_dict.items():
            parts = k.split('.')
            try:
                #  Extract core module name
                if 'encoder' in parts:
                    idx = parts.index('encoder')
                elif 'neck' in parts:
                    idx = parts.index('neck')
                elif 'head' in parts:
                    idx = parts.index('head')
                else:
                    continue

                new_key = ".".join(parts[idx:])

                # Check if key exists in current model
                if new_key in model_state_dict:
                    # Check if shape matches
                    if v.shape != model_state_dict[new_key].shape:
                        print(f"  [Jump] Skip parameter {new_key}: Shape mismatch "
                              f"(Src: {list(v.shape)} vs Dst: {list(model_state_dict[new_key].shape)})")
                        continue  # Skip directly if shape mismatch (e.g., different Head class numbers)

                    new_sd[new_key] = v
            except ValueError:
                continue

        # Load filtered weights
        msg = self.models[task_id].load_state_dict(new_sd, strict=strict)  # Must set to False because Head is skipped

        loaded_count = len(new_sd)
        print(f"  [*] Successfully extracted and loaded {loaded_count} parameter blocks to {task_id}")
        if len(msg.missing_keys) > 0:
            # At this time, Head-related Keys should appear in missing_keys, which is normal
            print(f"  [Info] Number of unmatched layers: {len(msg.missing_keys)} ")

        return msg


    def load_shared_multitask_weights(self, weight_path: str, strict: bool = False):
        """
        Specifically used to load mixed weights of 1 Backbone + 27 task Heads
        """
        if not os.path.exists(weight_path):
            print(f"  [Error] Weight file does not exist: {weight_path}")
            return False

        print(f"  [*] Parsing multi-task shared weights: {os.path.basename(weight_path)}")
        state_dict = torch.load(weight_path, map_location='cpu')

        # Iterate through all instantiated sub-models (the 27 tasks) in the Factory
        for tid, sub_model in self.models.items():
            new_sd = {}
            model_sd = sub_model.state_dict()

            for k, v in state_dict.items():
                # Logic A: If weight Key contains current task ID, it means it's the Head of this task
                # Logic B: If weight Key contains 'encoder' or 'neck', it means it's the shared Backbone

                parts = k.split('.')
                # Extract standardized key (remove prefixes like models.task_id.)
                try:
                    if 'encoder' in parts:
                        new_key = ".".join(parts[parts.index('encoder'):])
                    elif 'neck' in parts:
                        new_key = ".".join(parts[parts.index('neck'):])
                    elif 'head' in parts and tid in k:  # Load only if current task_id is included in Head path
                        new_key = ".".join(parts[parts.index('head'):])
                    else:
                        continue

                    # Shape check: prevent crash caused by inconsistent number of classes
                    if new_key in model_sd and v.shape == model_sd[new_key].shape:
                        new_sd[new_key] = v
                except ValueError:
                    continue

            msg = sub_model.load_state_dict(new_sd, strict=strict)
            print(f"  [+] Task branch {tid} loaded successfully: Matched items{len(new_sd)}")
        return True
    def load_weights_to_branch(self, target_task_id: str, weight_path: str, strict: bool = False):
        """
        Core function to implement Warm Start and Assembler
        Function: Load the weights pointed to by weight_path into the target_task_id branch in this model.
        """
        if target_task_id not in self.models:
            print(f"  [Error] Branch {target_task_id} does not exist, cannot load weights.")
            return

        if not os.path.exists(weight_path):
            print(f"  [Error] Weight file does not exist: {weight_path}")
            return

        print(f"  [*] Loading weights: {os.path.basename(weight_path)} -> {target_task_id}")
        checkpoint = torch.load(weight_path, map_location='cpu')

        # Handle Key mapping logic of weight dictionary
        # Case A: Weight file is complete MultiTaskModelFactory saved by train.py (with models.task_id.xxx prefix)
        # Case B: Weight file is individually saved ATLASv57 (without models prefix)

        raw_state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        new_state_dict = {}

        prefix = f"models.{target_task_id}."  # Expected prefix

        # Automatically strip prefix to align with self.models[target_task_id]
        for k, v in raw_state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k.replace(prefix, "")] = v
            else:
                # Compatible with weights saved directly by single task
                new_state_dict[k] = v

        msg = self.models[target_task_id].load_state_dict(new_state_dict, strict=strict)
        print(f"  [Success] Loading result: {msg}")
        return msg