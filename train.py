import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tff import TFFModel
from tff.data import SkeletonDataset
from tff.cache import cached_topology
from tff.calibration import select_threshold


@torch.no_grad()
def evaluate(model, loader, device, prefix_frames=0):
    correct = filtered = total = accepted = 0
    confusion = torch.zeros(model.classes, model.classes, dtype=torch.long)
    for x, y, descriptor in loader:
        y = y.to(device)
        result = model.predict(x.to(device), descriptor=descriptor.to(device), prefix_frames=prefix_frames)
        prediction = result['prediction']
        correct += (result['logits'].argmax(-1) == y).sum().item()
        filtered += (prediction == y).sum().item()
        accepted += result['accepted'].sum().item()
        total += len(y)
        indices = (y * model.classes + prediction).cpu()
        confusion += torch.bincount(indices, minlength=model.classes ** 2).reshape(model.classes, model.classes)
    support = confusion.sum(-1)
    precision = confusion.diag() / confusion.sum(0).clamp_min(1)
    recall = confusion.diag() / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {'backbone_accuracy': correct / total, 'tff_accuracy': filtered / total,
            'macro_f1': f1[support > 0].mean().item(), 'acceptance_rate': accepted / total,
            'confusion_matrix': confusion.tolist()}


@torch.no_grad()
def calibrate(model, loader, device, thresholds):
    model.eval()
    scores, correct = [], []
    for x, labels, descriptor in loader:
        descriptor = descriptor.to(device)
        auxiliary = model.auxiliary(model.encode_topology(descriptor))
        candidate = auxiliary.argmax(-1)
        density = model.kde.log_density(descriptor)
        score = density if model.density_mode == 'density' else density.log_softmax(-1)
        contact = model.contacts.scores(model.contacts(x.to(device)))
        consistent = contact.gather(1, candidate[:, None]).squeeze(1) >= model.contact_threshold
        scores.append(score.gather(1, candidate[:, None]).squeeze(1).masked_fill(~consistent, -torch.inf).cpu())
        correct.append((candidate.cpu() == labels))
    log_grid = torch.tensor(thresholds).log().tolist()
    best, rows = select_threshold(torch.cat(scores), torch.cat(correct), log_grid)
    model.threshold = float(np.exp(best['threshold']))
    for row in rows:
        row['threshold'] = float(np.exp(row['threshold']))
    return {'threshold': model.threshold, 'grid': rows}


def arguments():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config')
    preset, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser(parents=[preliminary])
    parser.add_argument('--data', required=True)
    parser.add_argument('--labels')
    parser.add_argument('--val-data')
    parser.add_argument('--val-labels')
    parser.add_argument('--calibration-data')
    parser.add_argument('--calibration-labels')
    parser.add_argument('--calibration-split', default='val')
    parser.add_argument('--classes', type=int, choices=(60, 120), default=60)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--aux-weight', type=float, default=0.2)
    parser.add_argument('--milestones', type=int, nargs='+', default=[35, 45, 55])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--channels', type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bandwidth', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--kde-samples', type=int, default=256)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--verify-threshold', type=float, default=0.05)
    parser.add_argument('--contact-radius', type=float, default=0.35)
    parser.add_argument('--contact-temperature', type=float, default=0.25)
    parser.add_argument('--contact-threshold', type=float, default=0.3)
    parser.add_argument('--fusion-weights', type=float, nargs=3, default=[1.0, 0.5, 0.5])
    parser.add_argument('--density-mode', choices=('density', 'posterior'), default='density')
    parser.add_argument('--ph-frames', type=int, default=8)
    parser.add_argument('--ph-points', type=int, default=96)
    parser.add_argument('--time-scale', type=float, default=0.2)
    parser.add_argument('--prefix-frames', type=int, default=60)
    parser.add_argument('--threshold-grid', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='runs/tff')
    parser.add_argument('--resume')
    if preset.config:
        defaults = json.loads(Path(preset.config).read_text(encoding='utf-8-sig'))
        valid = {action.dest for action in parser._actions}
        if set(defaults) - valid:
            parser.error('Unknown configuration keys: ' + str(set(defaults) - valid))
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 55 if args.classes == 60 else 60
    if args.epochs < 1 or args.batch_size < 1 or any(t <= 0 for t in args.threshold_grid):
        parser.error('epochs, batch size and thresholds must be positive')
    return args


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(classes=args.classes, channels=args.channels, bandwidth=args.bandwidth,
                  threshold=args.threshold, max_samples=args.kde_samples, topk=args.topk,
                  density_mode=args.density_mode, ph_frames=args.ph_frames, ph_points=args.ph_points,
                  time_scale=args.time_scale, verify_threshold=args.verify_threshold,
                  fusion_weights=args.fusion_weights, dropout=args.dropout,
                  contact_radius=args.contact_radius, contact_temperature=args.contact_temperature,
                  contact_threshold=args.contact_threshold)
    model = TFFModel(**config)
    train_base = SkeletonDataset(args.data, 'train', args.frames, args.labels)
    if train_base.labels.min() < 0 or train_base.labels.max() >= args.classes:
        raise ValueError('Labels must be zero-based and smaller than --classes')
    train_set = cached_topology(train_base, model.topology, output / 'cache', args.data,
                               'train', args.batch_size)
    model.kde.fit(torch.as_tensor(np.array(train_set.topology, copy=True)),
                  torch.as_tensor(train_base.labels, dtype=torch.long), args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    contact_values, contact_labels = [], []
    with torch.no_grad():
        for x, labels, _ in DataLoader(train_set, batch_size=args.batch_size, shuffle=False):
            contact_values.append(model.contacts(x))
            contact_labels.append(labels)
        model.contacts.fit(torch.cat(contact_values), torch.cat(contact_labels))

    def make_loader(path, labels, split):
        base = SkeletonDataset(path, split, args.frames, labels)
        if base.labels.min() < 0 or base.labels.max() >= args.classes:
            raise ValueError('Invalid evaluation labels')
        dataset = cached_topology(base, model.topology, output / 'cache', path, split, args.batch_size)
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    val_loader = None
    if args.val_data or Path(args.data).suffix == '.npz':
        val_loader = make_loader(args.val_data or args.data, args.val_labels, 'val')
    calibration_loader = None
    if args.calibration_data:
        calibration_loader = make_loader(args.calibration_data, args.calibration_labels, args.calibration_split)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state['config'] != config:
            raise ValueError('Resume checkpoint model configuration does not match')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch']
        if 'rng_state' in state:
            torch.set_rng_state(state['rng_state'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng_state'):
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng_state']])
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = total = 0
        for x, y, descriptor in train_loader:
            x, y, descriptor = x.to(device), y.to(device), descriptor.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(x, descriptor)
            loss = F.cross_entropy(logits, y) + args.aux_weight * F.cross_entropy(auxiliary, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
        scheduler.step()
        print(json.dumps({'epoch': epoch + 1, 'loss': total_loss / total,
                          'lr': scheduler.get_last_lr()[0]}), flush=True)
        torch.save({'model': model.state_dict(), 'config': config, 'frames': args.frames,
                    'epoch': epoch + 1, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   output / 'last.pt')
    if calibration_loader is not None:
        calibration = calibrate(model, calibration_loader, device, args.threshold_grid)
        config['threshold'] = model.threshold
        (output / 'calibration.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    torch.save({'model': model.state_dict(), 'config': config, 'frames': args.frames,
                'epoch': args.epochs}, output / 'model.pt')
    if val_loader is not None:
        metrics = evaluate(model, val_loader, device, args.prefix_frames)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in metrics.items() if k != 'confusion_matrix'}))


if __name__ == '__main__':
    main()
