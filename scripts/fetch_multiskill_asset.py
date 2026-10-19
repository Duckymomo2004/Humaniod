"""Download only the official G1 29 DoF USD asset, without overwriting existing files."""

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

REVISION = "323e350252b9c3aee9c40acbfdad84f6ce46a5ac"
DATASET = "unitreerobotics/unitree_model"
ASSET_DIR = "G1/29dof/usd/g1_29dof_rev_1_0"
ROOT = Path(__file__).resolve().parents[1]
destination = Path(os.environ.get("UNITREE_MODEL_DIR", ROOT / ".deps/unitree_model-multiskill"))

with urlopen(f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{ASSET_DIR}?recursive=true", timeout=60) as response:
    entries = json.load(response)

manifest = {"dataset": DATASET, "revision": REVISION, "files": []}
for entry in entries:
    if entry["type"] != "file":
        continue
    relative_path = Path(entry["path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Invalid asset path: {relative_path}")
    target = destination / relative_path
    if target.exists():
        data = target.read_bytes()
    else:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
    assert len(data) == entry["size"], str(target)
    digest = hashlib.sha256(data).hexdigest()
    if "lfs" in entry:
        assert digest == entry["lfs"]["oid"], str(target)
    else:
        git_digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert git_digest == entry["oid"], str(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as asset_file:
            asset_file.write(data)
    manifest["files"].append({"path": entry["path"], "size": len(data), "sha256": digest})
    print(f"[VERIFIED] {target} ({len(data)} bytes)", flush=True)

manifest_path = destination / "g1_29dof_manifest.json"
if not manifest_path.exists():
    with manifest_path.open("x") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
print(f"[PASS] Official G1 29 DoF asset at revision {REVISION}")
