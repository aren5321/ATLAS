from task_manager import TaskManager
from deploy import print_logo
import os
def interactive_register():
    tm = TaskManager()

    tm.list_tasks()
    print("-" * 30)

    # 2. Get Task ID and check for duplication
    while True:
        task_id = input("\nPlease enter a unique Task ID (e.g. thyroid_nodule_seg): ").strip()
        if not task_id:
            print("  [Error] Task ID cannot be empty!")
            continue
        if task_id in tm.db:
            print(f"  [Warning] Task ID '{task_id}' already exists. Please choose another name or delete the old task first.")
            continue
        break

    # 3. Detailed description (Description)
    description = input("Please enter a detailed task description (e.g. thyroid nodule contour segmentation v1): ").strip()
    if not description:
        description = "No description"

    # 4. Select anatomical site
    anatomies = ["Fetal", "Breast", "Thyroid", "Liver", "Heart", "Lung", "Abdomen","ovary","cervix", "Other"]
    print("\nAvailable anatomy list:")
    for i, a in enumerate(anatomies):
        print(f" [{i}] {a}")

    try:
        a_idx = int(input("Please select anatomy index: "))
        anatomy = anatomies[a_idx]
    except (ValueError, IndexError):
        print("  [Info] Invalid index input, defaulting to 'Other'")
        anatomy = "Other"

    if anatomy == "Other":
        anatomy = input("Please enter a custom anatomy name: ").strip() or "Other"

    # 5. Task type
    types = ["classification", "segmentation", "regression", "detection"]
    print("\nTask type list:")
    for i, t in enumerate(types):
        print(f" [{i}] {t}")

    try:
        t_idx = int(input("Please select task type index: "))
        task_name = types[t_idx]
    except (ValueError, IndexError):
        task_name = "classification"  # default

    # 6. Number of classes and labels
    try:
        num_classes = int(input("\nPlease enter number of classes (num_classes): "))
    except ValueError:
        num_classes = 1

    label_map = {}
    print(f"--- Recording clinical meaning for {num_classes} classes ---")
    for i in range(num_classes):
        meaning = input(f"  Clinical meaning for ID {i}: ").strip()
        label_map[str(i)] = meaning or f"Class_{i}"

    # 7. Dataset
    ds_name = input("\nPlease enter dataset name (default FMC_UIA): ") or "FMC_UIA"
    ds_types = ["Public", "Private"]
    print("Dataset source:")
    for i, d in enumerate(ds_types): print(f" [{i}] {d}")
    try:
        d_idx = int(input("Please select index: "))
        dataset_type = ds_types[d_idx]
    except (ValueError, IndexError):
        dataset_type = "Private"

    # 8. Call Manager to save into database
    # Note: TaskManager.register_task should support description
    tm.register_task(
        task_id=task_id,
        anatomy=anatomy,
        task_name=task_name,
        num_classes=num_classes,
        label_map=label_map,
        dataset_name=ds_name,
        dataset_type=dataset_type,
        description=description  # pass to manager
    )

    print(f"\n" + "*" * 40)
    print(f" [Success] Task '{task_id}' has been registered!")
    print(f" Description: {description}")
    print("\n[Info] After training finishes and before committing weights and task_db.json, please run git pull first. If there is any task_id conflict, please run merge_db.py.")
    print("*" * 40 + "\n")


if __name__ == "__main__":
    print_logo()
    interactive_register()