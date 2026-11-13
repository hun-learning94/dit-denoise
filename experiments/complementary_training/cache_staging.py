"""Hydrate an allocation-local cache from a verified preprocessed copy."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time


def hydrate_cache(project, destination, identity):
    """Called with the destination writer lock held; never change pixel content."""
    source = Path(project) / 'datasets' / 'complementary_pixel_cache' / destination.name
    staged = source / 'STAGED.json'
    ready = source / 'READY.json'
    if not staged.exists() or not ready.exists():
        return False
    manifest = json.loads(staged.read_text())
    record = json.loads(ready.read_text())
    assert manifest['completed'] and manifest['identity'] == record['identity'] == identity
    data = source / 'images.npy'
    assert data.stat().st_size == record['file_bytes'] == manifest['file_bytes']
    if shutil.disk_usage(destination).free < data.stat().st_size + 2*2**30:
        return False
    temporary = destination / '.images.hydrate.tmp'
    digest = hashlib.sha256()
    started, last, copied = time.monotonic(), time.monotonic(), 0
    try:
        with data.open('rb') as stream, temporary.open('wb') as target:
            while block := stream.read(8*2**20):
                digest.update(block)
                target.write(block)
                copied += len(block)
                if time.monotonic()-last > 20:
                    print('CACHE HYDRATE', destination.name, copied, manifest['file_bytes'], flush=True)
                    last = time.monotonic()
            target.flush()
            os.fsync(target.fileno())
        assert copied == manifest['file_bytes'] and digest.hexdigest() == manifest['sha256']
        temporary.replace(destination / 'images.npy')
        record.update(hydrated_from_shared=str(source), hydrated_sha256=digest.hexdigest(),
            hydrated_seconds=time.monotonic()-started, hydrated_host=socket.gethostname())
        ready_tmp = destination / '.READY.hydrate.tmp'
        ready_tmp.write_text(json.dumps(record, indent=2)+'\n')
        ready_tmp.replace(destination / 'READY.json')
        print('CACHE HYDRATE COMPLETE', destination.name, copied, record['hydrated_seconds'], flush=True)
        return True
    finally:
        if temporary.exists(): temporary.unlink()
