import os
import shutil
from datetime import datetime
from typing import List, Optional
import json


def archive_output(src_paths: List[str], step_name: str, archive_root: str = 'archive', extra_note: Optional[str] = None) -> str:
    """
    Archive files or directories to a timestamped archive directory for a given pipeline step.
    Also creates a manifest.json with metadata about the archive.

    Args:
        src_paths (List[str]): List of file or directory paths to archive
        step_name (str): Name of the pipeline step (e.g., 'embeddings', 'clustering')
        archive_root (str): Root directory for all archives
        extra_note (Optional[str]): Optional string to append to the archive folder name

    Returns:
        str: Path to the created archive directory
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    note = f"_{extra_note}" if extra_note else ""
    archive_dir = os.path.join(archive_root, step_name, f"{timestamp}{note}")
    os.makedirs(archive_dir, exist_ok=True)

    archived_files = []
    for src in src_paths:
        if not os.path.exists(src):
            print(f"[archive_output] Warning: {src} does not exist, skipping.")
            continue
        base_name = os.path.basename(src.rstrip('/'))
        dest = os.path.join(archive_dir, base_name)
        if os.path.isdir(src):
            shutil.copytree(src, dest)
            archived_files.append({'type': 'directory', 'src': src, 'dest': dest})
        else:
            shutil.copy2(src, dest)
            archived_files.append({'type': 'file', 'src': src, 'dest': dest})

    # Write manifest.json
    manifest = {
        'step_name': step_name,
        'timestamp': timestamp,
        'extra_note': extra_note,
        'archived_files': archived_files
    }
    with open(os.path.join(archive_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[archive_output] Archived to {archive_dir}")
    return archive_dir 