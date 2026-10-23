from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess(array, frames=300):
    """C,T,V,M -> C,frames,V,M. Translate each frame to spine base."""
    x = np.array(array, dtype=np.float32, copy=True)
    if x.ndim != 4 or x.shape[0] != 3 or x.shape[2] != 25 or x.shape[3] < 1:
        raise ValueError("Expected skeleton shape (3,T,25,M)")
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    valid_frames = np.flatnonzero(np.any(x != 0, axis=(0, 2, 3)))
    if not len(valid_frames):
        return torch.zeros((3, frames, 25, x.shape[3]), dtype=torch.float32)
    x = x[:, valid_frames[0]:valid_frames[-1] + 1]
    valid_joints = np.any(x != 0, axis=0, keepdims=True)
    # A shared origin preserves the relative positions of interacting people.
    person_valid = valid_joints.any(axis=(0, 2))
    primary = person_valid.argmax(axis=-1)
    root = x[:, np.arange(x.shape[1]), 0, primary][:, :, None, None].copy()
    x = np.where(valid_joints, x - root, 0)
    if x.shape[1] < frames:
        x = np.tile(x, (1, int(np.ceil(frames / x.shape[1])), 1, 1))[:, :frames]
    else:
        index = np.linspace(0, x.shape[1] - 1, frames).round().astype(int)
        x = x[:, index]
    return torch.from_numpy(np.ascontiguousarray(x))


class SkeletonDataset(Dataset):
    """NPZ: x_train,y_train,x_val,y_val; NPY: paired data and labels."""
    def __init__(self, path, split="train", frames=300, labels=None):
        if frames < 1:
            raise ValueError("frames must be positive")
        self.frames = frames
        if Path(path).suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                self.data = archive[f"x_{split}"]
                self.labels = archive[f"y_{split}"]
        else:
            self.data = np.load(path, mmap_mode="r", allow_pickle=False)
            if labels is None:
                raise ValueError("NPY data requires --labels")
            self.labels = np.load(labels, allow_pickle=False)
        if self.labels.ndim == 2:
            self.labels = self.labels.argmax(-1)
        if self.labels.ndim != 1 or not np.isfinite(self.labels).all() or not np.equal(self.labels, np.floor(self.labels)).all():
            raise ValueError("Labels must be finite integer class indices")
        self.labels = self.labels.astype(np.int64, copy=False)
        if self.data.ndim != 5 or self.data.shape[1] != 3 or self.data.shape[3] != 25:
            raise ValueError("Data must have shape (N,3,T,25,M)")
        if len(self.data) != len(self.labels) or len(self.data) == 0:
            raise ValueError("Dataset is empty or label count is incorrect")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return preprocess(self.data[index], self.frames), int(self.labels[index])
