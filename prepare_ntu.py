"""Convert NTU .skeleton files into protocol-separated NPY arrays."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from tff.data import preprocess


TRAIN_SUBJECTS_60 = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
TRAIN_SUBJECTS_120 = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                      31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                      58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                      93, 94, 95, 97, 98, 100, 103, 106}


def read_skeleton(path, persons=2):
    tracks = {}
    with open(path, encoding="utf-8-sig") as stream:
        frames = int(stream.readline())
        if frames < 1:
            raise ValueError(f"Empty sequence: {path}")
        for frame in range(frames):
            for _ in range(int(stream.readline())):
                identifier = stream.readline().split()[0]
                count = int(stream.readline())
                coordinates = np.array([[float(v) for v in stream.readline().split()[:3]]
                                        for _ in range(count)], dtype=np.float32)
                if count != 25:
                    raise ValueError(f"Expected 25 joints: {path}")
                tracks.setdefault(identifier, np.zeros((3, frames, 25), dtype=np.float32))
                tracks[identifier][:, frame] = coordinates.T
    if not tracks:
        raise ValueError(f"No skeleton tracks: {path}")
    # Stable IDs preserve person identity across frames.
    ordered = sorted(tracks.items(), key=lambda item: (-np.any(item[1] != 0, axis=(0, 2)).sum(), item[0]))
    data = np.zeros((3, frames, 25, persons), dtype=np.float32)
    for index, (_, track) in enumerate(ordered[:persons]):
        data[..., index] = track
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", type=int, choices=(60, 120), default=60)
    parser.add_argument("--protocol", choices=("xsub", "xview", "xset"), default="xsub")
    parser.add_argument("--missing-list", help="Official missing-skeleton list")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--persons", type=int, default=2)
    args = parser.parse_args()
    if (args.classes == 60 and args.protocol == "xset") or (args.classes == 120 and args.protocol == "xview"):
        parser.error("NTU60 supports xsub/xview; NTU120 supports xsub/xset")
    excluded = set()
    if args.missing_list:
        excluded = {Path(line.strip()).stem for line in Path(args.missing_list).read_text().splitlines() if line.strip()}
    pattern = re.compile(r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)")
    splits = {"train": [], "val": []}
    subjects = TRAIN_SUBJECTS_60 if args.classes == 60 else TRAIN_SUBJECTS_120
    for path in sorted(Path(args.input).rglob("*.skeleton")):
        match = pattern.fullmatch(path.stem)
        if not match or path.stem in excluded:
            continue
        setup, camera, subject, _, action = map(int, match.groups())
        if action > args.classes or (args.classes == 60 and setup > 17):
            continue
        train = (subject in subjects if args.protocol == "xsub" else
                 camera in (2, 3) if args.protocol == "xview" else setup % 2 == 0)
        splits["train" if train else "val"].append((path, action - 1))
    if not all(splits.values()):
        raise ValueError("Both protocol partitions must contain skeleton files")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"classes": args.classes, "protocol": args.protocol, "splits": {}}
    for split, items in splits.items():
        array = np.lib.format.open_memmap(output / f"{split}.npy", mode="w+", dtype="float32",
                                          shape=(len(items), 3, args.frames, 25, args.persons))
        labels, names = [], []
        for index, (path, label) in enumerate(items):
            array[index] = preprocess(read_skeleton(path, args.persons), args.frames).numpy()
            labels.append(label)
            names.append(path.name)
            if (index + 1) % 100 == 0 or index + 1 == len(items):
                print(f"{split}: {index + 1}/{len(items)}", flush=True)
        array.flush()
        del array
        np.save(output / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
        manifest["splits"][split] = names
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
