from pathlib import Path

from dirsync import sync


def sync_from_cloud_if_needed(remote_path: Path, local_path: Path, run_type: str) -> bool:
    """If running in cloud, sync data from network mount to local storage first."""
    if run_type == 'cloud':
        print(f"Cloud run detected. Syncing data from {remote_path} to {local_path}...")
        local_path.mkdir(parents=True, exist_ok=True)
        sync(str(remote_path), str(local_path), 'sync', only_newer=True, verbose=True)
        print(f"Dataset sync complete.")
        return True
    return False


def sync_to_cloud_if_needed(local_path: Path, remote_path: Path, run_type: str) -> bool:
    """If running in cloud, sync trained models from local storage to network mount."""
    if run_type == 'cloud':
        print(f"Cloud run detected. Syncing trained models from {local_path} to {remote_path}...")
        remote_path.mkdir(parents=True, exist_ok=True)
        sync(str(local_path), str(remote_path), 'sync', only_newer=True, verbose=True)
        print(f"Trained models sync complete.")
        return True
    return False
