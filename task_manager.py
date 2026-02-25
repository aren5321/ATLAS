import json
import os
from datetime import datetime
import pandas as pd

class TaskManager:
    def __init__(self, db_path='tasks_db.json'):
        self.db_path = db_path
        self.db = self._load_db()

    def _load_db(self):
        if not os.path.exists(self.db_path):
            return {}
        with open(self.db_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def save(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, indent=4, ensure_ascii=False)

    def register_task(self, task_id, anatomy, task_name, num_classes, label_map, dataset_name, dataset_type,description=""):
        """Register a new task"""
        self.db[task_id] = {
            "anatomy": anatomy,
            "task_name": task_name, # regression, segmentation, classification, detection
            "num_classes": num_classes,
            "label_map": label_map,
            "dataset": {
                "name": dataset_name,
                "type": dataset_type # Public or Private
            },
            "status": "pending",
            "weight_path": f"weights/{task_id}.pth",
            "best_metrics": {},
            "description": description,
            "register_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save()

    def get_task_config(self, task_id):
        """Return task config for training and inference scripts"""
        if task_id not in self.db:
            raise ValueError(f"Task ID '{task_id}' is not registered in the database.")
        return self.db[task_id]

    def get_all_tasks(self):
        """
        Return a list of configs for all registered tasks.
        Automatically inject dictionary key into config as 'task_id' field.
        """
        all_configs = []
        for task_id, config in self.db.items():
            # Shallow copy to avoid mutating original db and inject task_id
            item = config.copy()
            item['task_id'] = task_id
            all_configs.append(item)
        return all_configs

    def update_training_result(self, task_id, metrics, status="completed"):
        """Update metrics and status after training is finished"""
        if task_id in self.db:
            self.db[task_id]["status"] = status
            self.db[task_id]["best_metrics"] = metrics
            self.db[task_id]["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save()
            print(f"Task {task_id} metrics have been updated.")

    def list_tasks(self):
        """
        List all tasks in a formatted table.
        Column order: Anatomy, Task ID, Task Type, Best Score.
        """
        if not self.db:
            print("\n[!] The database is empty. No tasks found.")
            return

        table_data = []
        for tid, info in self.db.items():
            # 1. Process metrics: extract first metric name and value
            metrics = info.get('best_metrics', {})
            score_display = "N/A"
            if metrics:
                metric_name = list(metrics.keys())[0]
                metric_val = list(metrics.values())[0]
                score_display = f"{metric_name}: {metric_val:.4f}"

            # 2. Build row data
            table_data.append({
                "Anatomy": info.get('anatomy', 'N/A').upper(),
                "Task ID": tid,
                "Task Type": info.get('task_name', 'N/A').capitalize(),
                "Best Score": score_display
            })

        # 3. Convert to DataFrame
        df = pd.DataFrame(table_data)

        # 4. Sort by anatomy then task type for easier inspection
        df = df.sort_values(by=['Anatomy', 'Task Type'])

        # 5. Print results
        print("\n" + "=" * 40)
        print(f"{'ATLAS TASK REPOSITORY SUMMARY':^45}")
        print("=" * 40)
        # justify='left' for left-aligned text, index=False to hide index column
        print(df.to_string(index=False, justify='left'))
        print("=" * 45 + "\n")

    def get_task_detail(self, task_id):
        """
        Show detailed information for a given task_id, with error handling and formatted output.
        """
        # 1. Robust check: ensure task exists
        if not task_id:
            print("\n[!] Error: Task ID cannot be empty.")
            return

        if task_id not in self.db:
            print(f"\n[!] Error: Task ID '{task_id}' not found in the database.")
            # Fuzzy suggestions: if user types wrong id, suggest similar ones
            suggestions = [tid for tid in self.db.keys() if task_id.lower() in tid.lower()]
            if suggestions:
                print(f"    Did you mean: {', '.join(suggestions)}?")
            return

        info = self.db[task_id]

        # 2. Nicely format and print details
        print("\n" + "·" * 50)
        print(f"{'DETAILED TASK CONFIGURATION':^50}")
        print("·" * 50)

        # Core metadata
        print(f"  > [BASIC INFO]")
        print(f"    - Task ID:      {task_id}")
        print(f"    - Anatomy:      {info.get('anatomy', 'N/A').upper()}")
        print(f"    - Task Type:    {info.get('task_name', 'N/A').upper()}")
        print(f"    - Description:  {info.get('description', 'No description provided.')}")

        # Training / model status
        print(f"\n  > [MODEL & STATUS]")
        print(f"    - Status:       {info.get('status', 'N/A').upper()}")
        print(f"    - Num Classes:  {info.get('num_classes', 'N/A')}")
        print(f"    - Label Map:    {info.get('label_map', 'N/A')}")
        print(f"    - Weight Path:  {info.get('weight_path', 'N/A')}")

        # Dataset details
        ds = info.get('dataset', {})
        print(f"\n  > [DATASET SOURCE]")
        print(f"    - Name:         {ds.get('name', 'N/A')}")
        print(f"    - Access Type:  {ds.get('type', 'N/A')}")

        #  Data fingerprint information
        print(f"\n  > [DATA FINGERPRINT]")
        print(f"    - Resample Target: {info.get('resample_target', 'Pending Analysis')}")
        print(f"    - Train Samples:   {info.get('train_samples', 0)} cases")

        # Historical metrics
        metrics = info.get('best_metrics', {})
        print(f"\n  > [PERFORMANCE METRICS]")
        if metrics:
            for k, v in metrics.items():
                print(f"    - {k:<12}: {v:.4f}")
        else:
            print("    - No metrics recorded yet.")

        # Timestamps
        print(f"\n  > [TIMESTAMPS]")
        print(f"    - Registered:   {info.get('register_time', 'N/A')}")
        if "last_update" in info:
            print(f"    - Last Updated: {info.get('last_update', 'N/A')}")

        print("·" * 50 + "\n")
