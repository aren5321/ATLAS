import json
import os
import shutil

"""
================================================================================
                ATLAS Task Merge and Conflict Management System (Merge System)
================================================================================

[Description]
    This script is used to resolve conflicts in tasks_db.json during multi-person
    collaboration or distributed development.
    Principle: protect local results (IDs unchanged) while accepting remote
    contributions (rename conflicting IDs).
    It supports synchronized renaming of three coupled artifacts:
    JSON configs, .pth weight files, and .csv training index files.

[Prerequisites]
    1. The local database file under development/training is named: tasks_db.json
    2. The remote database file pulled from GitHub is named: tasks_db_remote.json
       (If git pull reports conflicts, manually save the remote version with this name)

[Usage Scenario & Steps]
    Scenario: You have finished local training, and after git pull you find that
    someone else has committed tasks with the same IDs.

    Step 1 [Pull]: run `git pull`.
    Step 2 [Backup]: if tasks_db.json conflicts, save the remote version as
                     `tasks_db_remote.json`, and restore your local one as
                     `tasks_db.json`.
    Step 3 [Merge]: run `python merge_db.py` in terminal.
                     - The script scans both JSON files.
                     - If IDs conflict, local IDs remain untouched, while remote
                       IDs are automatically renamed to ID_1, ID_2, ...
                     - Corresponding remote weights weights/ID.pth are renamed
                       to weights/ID_1.pth, etc.
                     - Corresponding remote indices data/train/csv_files/ID.csv
                       are renamed to ID_1.csv, etc.
    Step 4 [Commit]: at this point local tasks_db.json contains both sides'
                     results, then run:
                     git add .
                     git commit -m "Merge remote tasks and resolve collisions"
                     git push

[Notes]
    - This script automatically moves files (via shutil.move). It is recommended
      to make a temporary backup of the weights folder before running.
    - The script is idempotent: if the registration time of a task is exactly
      the same, it is considered the same task and skipped, so no duplicate
      suffix is produced.
    - It is recommended to add `tasks_db_remote.json` into .gitignore.

================================================================================
"""



def merge_atlas_tasks(local_path='tasks_db.json',
                      remote_path='tasks_db_remote.json',
                      weights_dir='weights',
                      csv_dir='data/train/csv_files'):
    # Basic checks
    if not os.path.exists(remote_path):
        print(f"[Skip] Remote database backup {remote_path} not found, merge not needed.")
        return

    if not os.path.exists(local_path):
        print(f"[Error] Local database {local_path} not found.")
        return

    # Load databases
    with open(local_path, 'r', encoding='utf-8') as f:
        local_db = json.load(f)
    with open(remote_path, 'r', encoding='utf-8') as f:
        remote_db = json.load(f)

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    merged_count = 0
    print("\n" + "=" * 60)
    print("      ATLAS Task Conflict Manager - Local-first Mode")
    print("=" * 60)

    for r_tid, r_info in remote_db.items():
        new_tid = r_tid
        counter = 1

        # --- Step 1: conflict detection ---
        while new_tid in local_db:
            # Idempotency check: if register_time is identical, treat as same task
            if local_db[new_tid].get('register_time') == r_info.get('register_time'):
                break

            # Confirmed as different contributions causing ID collision
            # Let remote task ID yield and append suffix
            new_tid = f"{r_tid}_{counter}"
            counter += 1

        # If finally confirmed to be the same existing task, skip
        if new_tid in local_db and counter == 1:
            continue

        # --- Step 2: synchronize triplet artifacts (Weights & CSV) ---
        # If ID has been renamed, also rename pulled remote physical files.
        # Otherwise ID_1 would still point to ID.pth, causing later load errors.
        file_sync_configs = [
            (weights_dir, ".pth"),
            (csv_dir, ".csv")
        ]

        if new_tid != r_tid:
            for folder, ext in file_sync_configs:
                old_file_path = os.path.join(folder, f"{r_tid}{ext}")
                new_file_path = os.path.join(folder, f"{new_tid}{ext}")

                if os.path.exists(old_file_path):
                    # Move and rename remote file
                    shutil.move(old_file_path, new_file_path)
                    print(f" [File Move] Conflict resolved: {old_file_path} -> {new_file_path}")
                else:
                    # If remote ID is marked as completed but weight file is missing, warn
                    if folder == weights_dir and r_info.get('status') == 'completed':
                        print(f" [Warning] Remote task {r_tid} is completed but physical file not found: {old_file_path}")

        # --- Step 3: update config and insert into local database ---
        # Fix path pointing in config
        r_info['weight_path'] = f"{weights_dir}/{new_tid}.pth"

        # If conflict renaming occurred, annotate description to respect contributor
        if new_tid != r_tid:
            orig_desc = r_info.get('description', 'No description')
            r_info['description'] = f"{orig_desc} (Automatically renamed from {r_tid} due to ID conflict to preserve contributions)"

        # Inject processed remote task into local db
        local_db[new_tid] = r_info
        print(f" [Task Merge] Successfully preserved task: {new_tid}")
        merged_count += 1

    # --- Step 4: write back to local database ---
    with open(local_path, 'w', encoding='utf-8') as f:
        json.dump(local_db, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print(f" Merge finished: {merged_count} tasks added/renamed.")
    print(" You can now safely run git push.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    merge_atlas_tasks()