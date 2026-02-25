from task_manager import TaskManager


def main():
    tm = TaskManager()

    # 1. First show concise list
    tm.list_tasks()

    # 2. Then loop and query detailed info
    while True:
        target_id = input("Please enter the Task ID to inspect (or 'q' to quit): ").strip()

        if target_id.lower() == 'q':
            print("Exit inspection.")
            break

        # Call the newly added function
        tm.get_task_detail(target_id)


if __name__ == "__main__":
    main()