"""Shared report location, including isolated replacement figure builds."""
import json
import os
from pathlib import Path

REPORT = Path(os.environ.get('EXPERIMENT0_REPORT', 'reports/experiment0'))


def imagenet_spec():
    manifest = json.loads((REPORT / 'allpatch_manifest.json').read_text())
    spec, = [s for s in manifest['models'] if s['dataset'] == 'imagenet64']
    return spec
