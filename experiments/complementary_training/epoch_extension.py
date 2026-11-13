"""Strictly validate an authorized 200-to-400 epoch continuation."""
import hashlib
import json


def recipe_hash(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def validate_resume(saved, current, current_hash, allow_extension=False):
    assert recipe_hash(current) == current_hash, 'Current recipe hash mismatch'
    previous = saved['recipe']
    assert recipe_hash(previous) == saved['recipe_sha256'], 'Checkpoint recipe hash mismatch'
    if saved['recipe_sha256'] == current_hash:
        return False
    assert allow_extension, 'Resume recipe or dataset changed'
    assert previous['epochs'] == 200 and current['epochs'] == 400, 'Unsupported epoch extension'
    assert previous['total_steps'] == 200 * previous['steps_per_epoch']
    assert current['total_steps'] == 400 * previous['steps_per_epoch']
    expected = dict(previous, epochs=400, total_steps=current['total_steps'])
    assert expected == current, 'Extension changed settings beyond epochs and total steps'
    assert 0 <= saved['step'] <= previous['total_steps'], 'Invalid parent checkpoint step'
    return True
