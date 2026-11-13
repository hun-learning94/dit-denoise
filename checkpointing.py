"""Atomically publish permanent training snapshots and a rolling latest state."""
import json
import os
from pathlib import Path
import tempfile

import torch


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_training_checkpoint(payload, archive_path=None, latest_path=None):
    """Keep archival files immutable when later rolling checkpoints replace latest."""
    assert archive_path is not None or latest_path is not None
    if archive_path is not None:
        archive_path = Path(archive_path)
        _atomic_write(archive_path, lambda handle: torch.save(payload, handle))
    if latest_path is None:
        return
    latest_path = Path(latest_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path is None:
        _atomic_write(latest_path, lambda handle: torch.save(payload, handle))
    else:
        # Same-directory hard links avoid serializing the same state twice.
        # Replacing latest later leaves the permanent snapshot's inode intact.
        with tempfile.TemporaryDirectory(dir=latest_path.parent, prefix='.latest-link-') as temporary:
            staged = Path(temporary) / latest_path.name
            os.link(archive_path, staged)
            os.replace(staged, latest_path)
    metadata = dict(checkpoint=latest_path.name, step=int(payload['step']),
                    bytes=latest_path.stat().st_size)
    _atomic_write(latest_path.with_suffix('.json'),
                  lambda handle: handle.write((json.dumps(metadata, indent=2) + '\n').encode()))
