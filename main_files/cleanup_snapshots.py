import os
import time

SNAPSHOT_DIR    = "/home/gayathri/Trivigi_Childsafety_system/main_files/snapshots"
MAX_AGE_SECONDS = 24 * 60 * 60


def cleanup():
    now     = time.time()
    removed = 0
    for name in os.listdir(SNAPSHOT_DIR):
        path = os.path.join(SNAPSHOT_DIR, name)
        if not os.path.isfile(path):
            continue
        if now - os.path.getmtime(path) > MAX_AGE_SECONDS:
            os.remove(path)
            removed += 1
    print(f"[INFO] Cleanup removed {removed} snapshot(s) older than 24h")


if __name__ == "__main__":
    cleanup()
