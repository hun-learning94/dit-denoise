"""Prepare pinned public checkpoints and native-resolution official validation data."""
from __future__ import annotations
from revision.runtime import path as _release_path

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/2026-11-06/attention_context_pilot"
WORK = Path(_release_path('assets/external/pixel-dit'))
SOURCE_DATA = Path(_release_path('assets/external/imagenet-val-2012'))

SPECS = {
    "pixnerd": {
        "github": "MCG-NJU/PixNerd", "source_revision": "6da060bd4b11a4a0ac2443a31b79701090e23c23",
        "repo": "MCG-NJU/PixNerd-XL-P16-C2I", "revision": "edd8adfd6b2f433a02543094f331c9207f40b9db",
        "filename": "epoch=319-step=1600000_emainit.ckpt", "bytes": 2816656156,
        "sha256": "8a3657d6dc4586950b207f6e1aed77c134a05a29a035a0953e0d39afbc98a967",
        "source_files": ["src/models/transformer/pixnerd_c2i.py", "configs_c2i/pix256std1_repa_pixnerd_xl.yaml",
                         "src/diffusion/flow_matching/scheduling.py", "src/diffusion/flow_matching/training_repa.py",
                         "src/data/dataset/imagenet.py", "src/models/autoencoder/pixel.py", "README.md"],
    },
    "pixeldit": {
        "github": "NVlabs/PixelDiT", "source_revision": "f7d7c35f71893d81c6e71d2576ea80601a1e8529",
        "repo": "nvidia/PixelDiT-ImageNet", "revision": "6584a95df021dd65e9e9257f671334c7bf1e4794",
        "filename": "imagenet256_pixeldit_xl_epoch320.ckpt", "bytes": 12801261470,
        "sha256": "e97a4fc5b3fe61ff733cea9a2ac0a1dd0c3956d46b13fd484e1c4bb862c10666",
        "source_files": ["pixdit_core/pixeldit_c2i.py", "pixdit_core/modules.py", "c2i/configs/pix256_xl.yaml",
                         "c2i/src/diffusion.py", "c2i/src/data.py", "README.md", "LICENSE"],
    },
}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sources():
    for name, spec in SPECS.items():
        base = REPORT / "upstream" / name
        records = {}
        for rel in spec["source_files"]:
            p = base / rel
            url = f"https://raw.githubusercontent.com/{spec['github']}/{spec['source_revision']}/{rel}"
            if not p.exists():
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    raise RuntimeError(f"Source fetch failed for {name}/{rel}: HTTP {r.status_code}")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r.content)
            records[rel] = {"sha256": sha256(p), "url": url}
        atomic_json(base / "SOURCE.json", {"revision": spec["source_revision"], "files": records})
    atomic_json(REPORT / "checkpoint_manifest.json", {
        name: {**spec, "path": str(WORK / "pretrained" / f"{name}_imagenet256" / spec["filename"])}
        for name, spec in SPECS.items()
    })


def download(name):
    s = SPECS[name]
    dest = WORK / "pretrained" / f"{name}_imagenet256" / s["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = REPORT / "downloads" / f"{name}.json"
    if dest.exists():
        if dest.stat().st_size != s["bytes"] or sha256(dest) != s["sha256"]:
            raise RuntimeError(f"Existing {name} checkpoint does not match publisher hash")
        print(name, "already verified", flush=True)
        return
    part = dest.with_suffix(dest.suffix + ".part")
    base_url = f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/{quote(s['filename'])}"
    started = time.time()
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        atomic_json(receipt, {"status": "downloading", "attempt": attempt + 1, "bytes": offset,
                              "url": base_url, "expected_sha256": s["sha256"]})
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            # Avoid reusing an expired cached signed redirect during a retry.
            with requests.get(base_url, params={"download": "true", "request_time": int(time.time())},
                              headers=headers, stream=True, timeout=(30, 180)) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"Checkpoint request returned HTTP {r.status_code}")
                append = bool(offset and r.status_code == 206)
                if append and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Unexpected resumed content range")
                with part.open("ab" if append else "wb") as f:
                    for b in r.iter_content(8 * 1024 * 1024):
                        if b:
                            f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
            if part.stat().st_size != s["bytes"]:
                raise RuntimeError(f"Incomplete download: {part.stat().st_size}/{s['bytes']}")
            digest = sha256(part)
            if digest != s["sha256"]:
                raise RuntimeError("Publisher SHA256 mismatch")
            part.replace(dest)
            record = {"status": "verified", "path": str(dest), "bytes": dest.stat().st_size,
                      "sha256": digest, "url": base_url, "seconds": time.time() - started,
                      "publisher_revision": s["revision"]}
            atomic_json(receipt, record)
            atomic_json(dest.parent / "download.json", record)
            print(name, "verified", record["bytes"], flush=True)
            return
        except Exception as e:
            # Never print redirects or authentication-bearing request objects.
            message = str(e) if isinstance(e, RuntimeError) else type(e).__name__
            print(name, "attempt", attempt + 1, message, flush=True)
            atomic_json(receipt, {"status": "retrying", "error": message, "attempt": attempt + 1})
    raise RuntimeError(f"Download failed: {name}; see {receipt}")


def prepare_data():
    import numpy as np
    from PIL import Image

    dest = WORK / "datasets/attention_context_imagenet256_20261106"
    ready = dest / "READY.json"
    if ready.exists():
        print("Data already prepared:", ready, flush=True)
        return
    provenance = json.loads((SOURCE_DATA / "complete.json").read_text())
    labels_path = SOURCE_DATA / "validation_synsets.txt"
    if not provenance["complete"] or sha256(labels_path) != provenance["labels_sha256"]:
        raise RuntimeError("Official validation provenance/label checksum failed")
    labels = labels_path.read_text().splitlines()
    classes = provenance["class_order"]
    if len(labels) != 50000 or classes != sorted(set(labels)) or len(classes) != 1000:
        raise RuntimeError("Unexpected official ImageNet label order/count")
    by_class = {w: [] for w in classes}
    for idx, w in enumerate(labels, 1):
        by_class[w].append(idx)
    if any(len(x) != 50 for x in by_class.values()):
        raise RuntimeError("Expected 50 official validation images per class")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "originals").mkdir(exist_ok=True)
    (dest / "crops").mkdir(exist_ok=True)
    used = set()
    excluded = []
    split_records = {}
    for split, count, seed in [("pilot", 16, 2026110601), ("calibration", 128, 2026110602), ("test", 256, 2026110603)]:
        rng = np.random.default_rng(seed)
        records = []
        for cls in rng.permutation(1000):
            wnid = classes[int(cls)]
            for idx in rng.permutation(by_class[wnid]):
                idx = int(idx)
                if idx in used:
                    continue
                filename = f"ILSVRC2012_val_{idx:08d}.JPEG"
                source = SOURCE_DATA / "images" / filename
                with Image.open(source) as im:
                    w, h = im.size
                    if min(w, h) < 256:
                        excluded.append({"split": split, "filename": filename, "width": w, "height": h,
                                         "reason": "native_short_side_below_256"})
                        continue
                    im = im.convert("RGB")
                    while min(im.size) >= 512:
                        im = im.resize(tuple(x // 2 for x in im.size), resample=Image.Resampling.BOX)
                    factor = 256 / min(im.size)
                    im = im.resize(tuple(round(x * factor) for x in im.size), resample=Image.Resampling.BICUBIC)
                    arr = np.asarray(im)
                    y0, x0 = (arr.shape[0] - 256) // 2, (arr.shape[1] - 256) // 2
                    crop = Image.fromarray(arr[y0:y0 + 256, x0:x0 + 256])
                    crop_path = dest / "crops" / filename.replace(".JPEG", ".png")
                    crop.save(crop_path)
                original = dest / "originals" / filename
                shutil.copyfile(source, original)  # natural new timestamps, no archive mtime restoration
                digest = sha256(original)
                if digest != sha256(source):
                    raise RuntimeError("Original-image copy mismatch")
                seed_i = seed + idx
                target_rng = np.random.default_rng(seed_i)
                central = [int(target_rng.integers(4, 12)), int(target_rng.integers(4, 12))]
                peripheral_candidates = [(r, c) for r in range(16) for c in range(16)
                                         if r in (0, 1, 14, 15) or c in (0, 1, 14, 15)]
                peripheral = list(peripheral_candidates[int(target_rng.integers(len(peripheral_candidates)))])
                records.append({"image_id": filename, "official_index": idx, "wnid": wnid,
                                "class_id": int(cls), "native_size": [w, h], "original": str(original),
                                "original_sha256": digest, "crop": str(crop_path), "crop_sha256": sha256(crop_path),
                                "targets": [central, peripheral], "split": split})
                used.add(idx)
                break
            if len(records) == count:
                break
        if len(records) != count:
            raise RuntimeError(f"Insufficient native images for {split}")
        split_records[split] = records
        atomic_json(dest / f"{split}.json", records)
        atomic_json(REPORT / "data" / f"{split}.json", records)
    all_records = sum(split_records.values(), [])
    if len({r['original_sha256'] for r in all_records}) != len(all_records):
        raise RuntimeError("Duplicate original bytes across selected observations")
    if len({r['crop_sha256'] for r in all_records}) != len(all_records):
        raise RuntimeError("Duplicate cropped images across selected observations")
    # Crops are lossless PNG, so compare actual RGB bytes too.
    pixel_hashes = []
    for r in all_records:
        with Image.open(r['crop']) as im:
            pixel_hashes.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    if len(set(pixel_hashes)) != len(pixel_hashes):
        raise RuntimeError("Cross-observation pixel duplicate")
    atomic_json(dest / "exclusions.json", excluded)
    atomic_json(REPORT / "data" / "exclusions.json", excluded)
    receipt = {"status": "ready", "source": str(SOURCE_DATA), "source_provenance": provenance,
               "split_counts": {k: len(v) for k, v in split_records.items()}, "selected": len(used),
               "native_resolution_exclusions": len(excluded), "upscaled": 0,
               "normalization": "RGB uint8 / 127.5 - 1", "crop": "ADM center crop, no native short side below256",
               "split_files": {k: str(dest / f"{k}.json") for k in split_records},
               "split_sha256": {k: sha256(dest / f"{k}.json") for k in split_records},
               "source_archive_verification": "Existing upstream receipt matches official ImageNet MD5; original tar not recopied",
               "selected_original_copies_hash_verified": True, "distinct_original_crop_and_pixel_hashes": True}
    atomic_json(ready, receipt)
    atomic_json(REPORT / "data" / "READY.json", receipt)
    print("Prepared validation subsets", receipt["split_counts"], "excluded",len(excluded),flush=True)


def parallel_pixeldit_download():
    """Resume the existing prefix using verified disjoint HTTP ranges."""
    name='pixeldit';s=SPECS[name]
    dest=WORK/'pretrained'/f'{name}_imagenet256'/s['filename']
    if dest.exists():
        download(name)
        return
    part=dest.with_suffix(dest.suffix+'.part')
    plan_path=dest.parent/'parallel_download_plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
    else:
        prefix=part.stat().st_size if part.exists() else 0
        span=(s['bytes']-prefix+5)//6
        plan={'prefix':prefix,'total':s['bytes'],'segments':[
            {'start':start,'end':min(start+span,s['bytes'])-1,'index':i}
            for i,start in enumerate(range(prefix,s['bytes'],span))]}
        atomic_json(plan_path,plan)
    base_url=f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/{quote(s['filename'])}"
    started=time.time()
    def fetch(segment):
        path=dest.with_suffix(f".segment{segment['index']}.part")
        length=segment['end']-segment['start']+1
        for attempt in range(4):
            have=path.stat().st_size if path.exists() else 0
            if have==length:return path
            if have>length:raise RuntimeError('Oversized partial range')
            start=segment['start']+have
            try:
                with requests.get(base_url,params={'download':'true','range_index':segment['index'],'now':int(time.time())},
                                  headers={'Range':f"bytes={start}-{segment['end']}"},stream=True,timeout=(30,180)) as r:
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f"bytes {start}-{segment['end']}/{s['bytes']}":
                        raise RuntimeError(f"Invalid range response HTTP {r.status_code}")
                    with path.open('ab') as f:
                        for b in r.iter_content(8*1024*1024):
                            if b:f.write(b)
                        f.flush();os.fsync(f.fileno())
                if path.stat().st_size!=length:raise RuntimeError('Incomplete range')
                print('pixeldit range completed',segment['index'],length,flush=True)
                return path
            except Exception as e:
                print('pixeldit range retry',segment['index'],attempt+1,type(e).__name__,flush=True)
        raise RuntimeError(f"Range {segment['index']} failed")
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths=list(pool.map(fetch,plan['segments']))
    with part.open('r+b' if part.exists() else 'w+b') as f:
        f.truncate(plan['prefix']);f.seek(plan['prefix'])
        for path in paths:
            with path.open('rb') as source:
                shutil.copyfileobj(source,f,8*1024*1024)
        f.flush();os.fsync(f.fileno())
    if part.stat().st_size!=s['bytes'] or sha256(part)!=s['sha256']:
        raise RuntimeError('Assembled checkpoint differs from publisher SHA256')
    part.replace(dest)
    receipt={'status':'verified','path':str(dest),'bytes':s['bytes'],'sha256':s['sha256'],
             'url':base_url,'publisher_revision':s['revision'],'seconds_parallel_resume':time.time()-started}
    atomic_json(REPORT/'downloads/pixeldit.json',receipt)
    atomic_json(dest.parent/'download.json',receipt)
    for path in paths:path.unlink()
    print('pixeldit verified publisher SHA256',s['sha256'],flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["sources", "download", "data", "download-parallel"])
    args = p.parse_args()
    if args.action == "sources":
        sources()
    elif args.action == "download":
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, SPECS))
    elif args.action=="download-parallel":
        parallel_pixeldit_download()
    else:
        prepare_data()


if __name__ == "__main__":
    main()
