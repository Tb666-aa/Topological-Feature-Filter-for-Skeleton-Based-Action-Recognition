import argparse
import json
import numpy as np
import torch
from tff import TFFModel
from tff.data import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPY: (3,T,25,M) or (N,3,T,25,M)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefix-frames", type=int, default=60)
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--output", default="predictions.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = TFFModel(**checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if data.ndim == 4:
        data = data[None]
    rows = []
    for start in range(0, len(data), args.batch_size):
        batch = torch.stack([preprocess(x, checkpoint["frames"])
                             for x in data[start:start + args.batch_size]]).to(args.device)
        result = model.predict(batch, not args.no_repeat, args.prefix_frames)
        for offset, prediction in enumerate(result["prediction"].tolist()):
            rows.append({"index": start + offset, "class_index": prediction,
                         "ntu_action_id": prediction + 1,
                         "accepted": bool(result["accepted"][offset]),
                         "stage": int(result["stage"][offset]),
                         "topology_candidate": int(result["candidate"][offset]),
                         "confidence": float(result["probability"][offset].max()),
                         "fusion_weights": result["fusion_weights"][offset].tolist(),
                         "rounds": int(result["rounds"][offset])})
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
