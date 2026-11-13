"""Keep one protected checkpoint selected by existing diagnostic measurements."""
import fcntl
import json
import math
import os
from pathlib import Path
import time


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not have completed its final JSONL line yet.
            continue
    return result


def protect_healthy_checkpoint(run_dir, max_ratio=1.25, gradient_limit=100.):
    root = Path(run_dir)
    diagnostics = [r for r in records(root / 'diagnostics.jsonl')
        if all(math.isfinite(float(r.get(k, float('nan')))) and r[k] >= 0
               for k in ('velocity_mse', 'denoising_mse'))]
    if not diagnostics:
        return None
    best = {k: min(r[k] for r in diagnostics) for k in ('velocity_mse', 'denoising_mse')}
    windows = records(root / 'optimization.jsonl')
    health = {r['step']: r for r in records(root / 'health.jsonl')}
    protection = root / 'protected'
    protection.mkdir(exist_ok=True)
    with (protection / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = protection / 'healthy.json'
        previous = json.loads(metadata.read_text()) if metadata.exists() else {}
        for row in sorted(diagnostics, key=lambda r: r['step'], reverse=True):
            if row['step'] <= previous.get('step', -1) and (protection / 'healthy.pt').exists():
                return previous
            if any(row[k] > max_ratio * best[k] for k in best):
                continue
            if health and not health.get(row['step'], {}).get('finite', False):
                continue
            recent = [r for r in windows if row['step']-1000 < r['step'] <= row['step']]
            if not recent or any(not math.isfinite(r['gradient_norm_max']) or
                                 r['gradient_norm_max'] > gradient_limit for r in recent):
                continue
            source = root / 'checkpoints' / f"step{row['step']:09d}.pt"
            temporary = protection / f'.healthy-{os.getpid()}.tmp'
            try:
                os.link(source, temporary)
            except FileNotFoundError:
                continue
            try:
                saved = temporary.stat()
                os.replace(temporary, protection / 'healthy.pt')
                record = dict(step=row['step'], checkpoint='protected/healthy.pt',
                    source_checkpoint=str(source), diagnostic=row, selection_best=best,
                    max_mse_ratio=max_ratio, gradient_norm_limit=gradient_limit,
                    device=saved.st_dev, inode=saved.st_ino, bytes=saved.st_size,
                    protected_unix_time=time.time(),
                    scope='Existing EMA diagnostics and finite-gradient history; no new evaluation')
                temp_json = metadata.with_suffix('.json.tmp')
                with temp_json.open('w') as stream:
                    json.dump(record, stream, indent=2)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_json, metadata)
                return record
            finally:
                if temporary.exists():
                    temporary.unlink()
        return previous or None
