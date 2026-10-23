# TFF Skeleton-Based Action Recognition

A PyTorch implementation of topology-aware skeleton action recognition with persistent homology, geometric descriptors, Gaussian KDE, learnable feature filtering, a multi-scale graph backbone, auxiliary verification and repeated screening.

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Architecture

1. Extract joint coordinates, parent-minus-child bone vectors, static angular encoding `1-cos(theta)`, and their frame differences. Smooth angular jumps greater than 45 degrees with a three-frame median filter. Concatenate the resulting 14 channels and project them to 64 channels.
2. Build a 10-dimensional sequence descriptor from geometry and Vietoris-Rips persistent homology over the field Z/2Z.
3. Standardize descriptors using training statistics. Fit class-conditional Gaussian KDE models with bandwidth 0.1. Encode descriptors into 64-dimensional learnable embeddings for auxiliary classification and channel gating.
4. Evaluate the auxiliary candidate against its class density. Verify contact consistency and search alternative candidates when the initial candidate fails either check. Only verified topology embeddings activate the feature gate.
5. Apply three spatiotemporal blocks. Each block contains exact-hop multi-scale spatial graph convolution with a learnable adjacency mask and seven temporal branches: four dilated convolutions, max pooling, average pooling and pointwise projection.
6. Fuse backbone, auxiliary and KDE distributions using entropy-dependent weights. For unresolved samples, average the fused distribution with a prediction from the first N frames.

## Persistent homology

`PersistentHomology` computes H0 and H1 persistence using Ripser. It samples up to eight frames and constructs a spatial point cloud from all valid joints and persons in each selected frame. It also constructs a four-dimensional spatiotemporal cloud by appending normalized time multiplied by 0.2. Deterministic farthest-point sampling caps each cloud at 96 points. Filtration uses Euclidean distance and runs through all edge distances. Infinite intervals and zero-lifetime intervals are excluded from summaries.

Persistence summaries are the average of the framewise summary and the spatiotemporal summary. These operations run on the CPU and are cached before training.

## Contact consistency

Six contact frequencies describe wrist-to-wrist, left/right wrist-to-head, left/right wrist-to-root, and cross-person wrist contact. Distances are divided by torso length and thresholded at 0.35. Each action class stores its mean training contact descriptor. Consistency is `exp(-mean_absolute_error / 0.25)`; initial and alternative candidates must exceed 0.3. Contact consistency also reweights the KDE distribution before decision fusion. Contact prototypes use training samples only.

## Ten-dimensional descriptor

All joint indices below are zero-based NTU indices.

| Index | Feature |
| --- | --- |
| 0 | Mean elbow and knee angle in radians |
| 1 | Standard deviation of elbow and knee angles |
| 2 | Mean distance across all non-adjacent joint pairs |
| 3 | Standard deviation of non-adjacent joint distances |
| 4 | Mean left/right upper-arm to forearm length ratio |
| 5 | Mean finite H0 lifetime |
| 6 | Maximum finite H0 lifetime |
| 7 | Sum of finite H1 lifetimes |
| 8 | Maximum finite H1 lifetime |
| 9 | H1 persistence entropy |

Angle triplets `(vertex, endpoint1, endpoint2)` are `(5,4,6)`, `(9,8,10)`, `(13,12,14)` and `(17,16,18)`. Each arm ratio divides the temporally summed upper-arm length by the temporally summed forearm length. Missing joints and absent persons are excluded from aggregation. A shared root translation preserves inter-person distances.

## Data preparation

Extract the NTU skeleton archives and provide the corresponding official missing-skeleton list.

```bash
python prepare_ntu.py --input data/nturgbd_skeletons --output data/ntu60_xsub --classes 60 --protocol xsub --missing-list data/NTU_RGBD_samples_with_missing_skeletons.txt
python prepare_ntu.py --input data/nturgbd_skeletons --output data/ntu120_xsub --classes 120 --protocol xsub --missing-list data/NTU_RGBD120_samples_with_missing_skeletons.txt
```

NTU60 supports `xsub` and `xview`; NTU120 supports `xsub` and `xset`. Body IDs are tracked across frames and the two tracks with the most valid frames are retained. The script writes `train.npy`, `train_labels.npy`, `val.npy`, `val_labels.npy` and a sample-name manifest. In these converted files, `val` denotes the official held-out protocol partition.

Data arrays have shape `(N,3,T,25,M)`, float32 coordinates and zero-based labels. Zero values represent missing joints, absent persons and padding. Short sequences are repeated to 300 frames; long sequences are uniformly subsampled. Standalone NPZ input is also supported with keys `x_train`, `y_train`, `x_val` and `y_val`.

## Training

```bash
python train.py --config configs/ntu60.json --data data/ntu60_xsub/train.npy --labels data/ntu60_xsub/train_labels.npy --val-data data/ntu60_xsub/val.npy --val-labels data/ntu60_xsub/val_labels.npy --output runs/ntu60
python train.py --config configs/ntu120.json --data data/ntu120_xsub/train.npy --labels data/ntu120_xsub/train_labels.npy --val-data data/ntu120_xsub/val.npy --val-labels data/ntu120_xsub/val_labels.npy --output runs/ntu120
```

For an NPZ file:

```bash
python train.py --config configs/ntu60.json --data data/ntu60.npz --output runs/ntu60
```

Configuration files contain the optimizer, schedule, model, persistence, filtering, fusion and reproducibility settings. Command-line arguments override configuration values.

| Setting | Default |
| --- | --- |
| Optimizer | SGD |
| Momentum | 0.9 |
| Initial learning rate | 0.1 |
| Weight decay | 0.0005 |
| Epochs | 55 for NTU60; 60 for NTU120 |
| Learning-rate milestones | 35, 45, 55 |
| Learning-rate multiplier | 0.1 |
| Batch size | 16 |
| Sequence frames | 300 |
| Block channels | 64, 128, 256 |
| Dropout | 0.1 |
| Auxiliary loss weight | 0.2 |
| KDE bandwidth | 0.1 |
| References per class | Up to 256 |
| Density threshold | 0.3 |
| Alternative candidates | Up to 5 |
| Auxiliary verification probability | 0.05 |
| Base fusion weights | 1.0, 0.5, 0.5 |
| Prefix frames | 60 |
| Random seed | 1 |

The loss is `CE(backbone_logits, label) + 0.2 * CE(auxiliary_logits, label)`. KDE references and descriptor normalization are fitted exclusively on the training split. Classes without training references receive zero KDE probability. KDE evaluates the full multivariate Gaussian product kernel in log space. Density thresholds operate in standardized descriptor space. Use `--density-mode posterior` to threshold the class-normalized KDE probability instead.

A calibration dataset can select the threshold by maximizing the F1 score of accepting correctly proposed auxiliary candidates. Keep it separate from the held-out protocol evaluation partition.

```bash
python train.py --config configs/ntu60.json --data data/train.npy --labels data/train_labels.npy --calibration-data data/calibration.npy --calibration-labels data/calibration_labels.npy --val-data data/test.npy --val-labels data/test_labels.npy --output runs/ntu60
```

Training saves `config.json`, `last.pt` with optimizer/scheduler/RNG states, and `model.pt` for inference. Optional outputs include `calibration.json` and `metrics.json` with backbone accuracy, TFF accuracy, macro F1, acceptance rate and the confusion matrix. Feature caches are stored in `cache/`; source metadata and implementation hashes invalidate stale entries.

Resume with the same configuration:

```bash
python train.py --config configs/ntu60.json --data data/ntu60.npz --output runs/ntu60 --resume runs/ntu60/last.pt
```

## Inference

```bash
python predict.py --checkpoint runs/ntu60/model.pt --input data/sample.npy --prefix-frames 60 --output predictions.json
```

Input shape is `(3,T,25,M)` or `(N,3,T,25,M)`. Output includes class indices, NTU action IDs, confidence, topology candidates, screening rounds, fusion weights and routing stage. Stages are 1 for initial acceptance, 2 for alternative acceptance and 3 for fallback; stage 0 indicates rejection with repeated screening disabled.

Use `--no-repeat` for initial filtering plus backbone classification without repeated screening or decision fusion. Use `--device cpu` for CPU execution.

Fusion weights are computed as `w_s * max(1 - entropy(p_s)/log(C), 1e-4)` and normalized across the three streams. Final probabilities are their weighted arithmetic average. Prefix fallback uses the same trained backbone and averages its prefix probability with the full-sequence fusion result.

## Verification

```bash
python -m unittest test_tff -v
```

Tests cover contact-state consistency, H1 detection on a circle, rigid-transform invariance, empty inputs, missing-person padding, inter-person geometry, analytic Gaussian KDE values, backpropagation, checkpoint round trips, alternative-candidate acceptance, prefix fallback and threshold selection.
