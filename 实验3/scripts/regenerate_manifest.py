#!/usr/bin/env python3
"""Regenerate SHA256 manifest for Experiment 4 handoff directory."""
import os, hashlib, csv

HANDOFF_DIR = r"E:\人参种在哪\实验4\00_input_from_experiment3"

manifest = []
for root, dirs, files in os.walk(HANDOFF_DIR):
    for f in sorted(files):
        fpath = os.path.join(root, f)
        sha = hashlib.sha256()
        with open(fpath, 'rb') as fh:
            for chunk in iter(lambda: fh.read(8192), b''):
                sha.update(chunk)
        rel = os.path.relpath(fpath, HANDOFF_DIR).replace('\\', '/')
        size = os.path.getsize(fpath)
        manifest.append({'relative_path': rel, 'sha256': sha.hexdigest(), 'size_bytes': size})

out_path = os.path.join(HANDOFF_DIR, 'sha256_manifest.csv')
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['relative_path', 'sha256', 'size_bytes'])
    w.writeheader()
    w.writerows(manifest)

print(f'Manifest regenerated with {len(manifest)} files:')
for m in manifest:
    print(f'  {m["relative_path"]}: {m["sha256"][:16]}... ({m["size_bytes"]:,} bytes)')
