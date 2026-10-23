import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CachedTopologyDataset(Dataset):
    def __init__(self, base, topology):
        self.base, self.topology = base, topology

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, label = self.base[index]
        return x, label, torch.from_numpy(np.array(self.topology[index], copy=True))


@torch.no_grad()
def cached_topology(dataset, extractor, directory, source, split, batch_size=16):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(source).resolve()
    persistence = extractor.persistence
    # Include implementation bytes so feature changes invalidate old caches.
    implementation = hashlib.sha256()
    for filename in ("features.py", "persistence.py", "data.py"):
        implementation.update(Path(__file__).with_name(filename).read_bytes())
    metadata = {"path": str(path), "size": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns, "split": split,
                "frames": dataset.frames, "samples": len(dataset),
                "ph_frames": persistence.sampled_frames, "ph_points": persistence.max_points,
                "time_scale": persistence.time_scale, "code": implementation.hexdigest()}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    target = directory / f"{split}_{key}.npy"
    if not target.exists():
        temporary = target.with_suffix(".partial.npy")
        result = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(dataset), 10))
        offset = 0
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            features = extractor(x).cpu().numpy()
            result[offset:offset + len(x)] = features
            offset += len(x)
            print(f"Topology cache {split}: {offset}/{len(dataset)}", flush=True)
        result.flush()
        del result
        temporary.replace(target)
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CachedTopologyDataset(dataset, np.load(target, mmap_mode="r"))
