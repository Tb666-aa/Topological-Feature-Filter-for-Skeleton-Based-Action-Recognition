import torch


@torch.no_grad()
def select_threshold(scores, correct, thresholds):
    """Maximize acceptance F1 using correctness labels from a calibration split."""
    best = None
    rows = []
    for threshold in thresholds:
        accepted = scores >= float(threshold)
        tp = (accepted & correct).sum().item()
        fp = (accepted & ~correct).sum().item()
        fn = (~accepted & correct).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        row = {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1}
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best = row
    if best is None:
        raise ValueError("Threshold grid must not be empty")
    return best, rows
