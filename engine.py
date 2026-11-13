"""Train / validate / test loop for CMA-VRP (mirrors the EASE Expert engine layout)."""
import numpy as np
import torch

from ease_bridge import Averager, calculate_metrics

try:  # EASE already uses tensorboardX; torch.utils.tensorboard would need `tensorboard`
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def make_writer(log_dir):
    if SummaryWriter is None:
        return None
    try:
        return SummaryWriter(log_dir)
    except Exception as e:  # noqa: BLE001
        print(f'[warn] TensorBoard writer disabled: {e}')
        return None


def _to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def train_one_epoch(train_loader, model, optimizer, scheduler, epoch, step, logger, config,
                    writer: SummaryWriter = None):
    losses = Averager()
    ys = Averager()
    cls = Averager()
    advs = Averager()
    parts = {}
    model.train()
    device = config.device_
    model.set_progress(epoch, config.epochs)

    for batch_idx, batch in enumerate(train_loader):
        batch = _to_device(batch, device)
        res = model(batch)

        loss = res['loss_total']
        if not torch.isfinite(loss):
            logger.warning('Non-finite loss at epoch %d step %d -> skipping batch', epoch, batch_idx)
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if not config.amp:
            scheduler.step()

        losses.add(loss.item())
        ys.add(res['loss_y'].item())
        cls.add(res['loss_cl'].item())
        advs.add(res['loss_adv'].item())
        for k in ('L_inter', 'L_intra', 'L_cycle', 'L_cycle_con', 'L_cycle_adv',
                  'L_ot', 'L_D0', 'L_D1', 'J'):
            parts.setdefault(k, Averager()).add(res[k].item())
        parts.setdefault('theta', Averager()).add(res['theta'].item())

        step += 1
        if batch_idx % config.print_interval == 0:
            logger.info(
                'Train Epoch: %d [%03d/%03d] loss %.4f (y %.4f cl %.4f adv %.4f) theta %.3f',
                epoch, batch_idx, len(train_loader), losses.item(), ys.item(),
                cls.item(), advs.item(), parts['theta'].item())
            if writer is not None:
                writer.add_scalar('train/loss', losses.item(), step)
                writer.add_scalar('train/loss_y', ys.item(), step)
                writer.add_scalar('train/loss_cl', cls.item(), step)
                writer.add_scalar('train/loss_adv', advs.item(), step)
                writer.add_scalar('train/theta', parts['theta'].item(), step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
                for k, v in parts.items():
                    writer.add_scalar(f'train/{k}', v.item(), step)
    return step


@torch.no_grad()
def val_one_epoch(test_loader, model, logger, config, epoch=0, writer: SummaryWriter = None):
    model.eval()
    device = config.device_
    preds, labels, losses = [], [], Averager()
    for batch in test_loader:
        batch = _to_device(batch, device)
        res = model(batch)
        losses.add(res['loss_total'].item())
        preds.append(res['classify_pred'].detach().cpu().numpy())
        labels.append(batch['y'].detach().cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    results = calculate_metrics(labels, preds, config.threshold)
    logger.info('Valid Loss: %.4f | acc %.4f mac_f1 %.4f auc %.4f',
                losses.item(), results['accuracy'], results['mac_f1'], results['auc'])
    if writer is not None:
        writer.add_scalar('val/loss', losses.item(), epoch)
        writer.add_scalar('val/acc', results['accuracy'], epoch)
        writer.add_scalar('val/mac_f1', results['mac_f1'], epoch)
    return results, losses.item()


@torch.no_grad()
def test_one_epoch(test_loader, model, logger, config, collect_features=False):
    model.eval()
    device = config.device_
    preds, labels, ids = [], [], []
    feats, flabels = [], []
    for batch in test_loader:
        batch = _to_device(batch, device)
        res = model(batch)
        preds.append(res['classify_pred'].detach().cpu().numpy())
        labels.append(batch['y'].detach().cpu().numpy())
        ids.extend(list(batch['id']))
        if collect_features:
            feats.append(res['final_feature'].detach().cpu().numpy())
            flabels.append(batch['y'].detach().cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    results = calculate_metrics(labels, preds, config.threshold)
    logger.info('Test: acc %.4f mac_f1 %.4f f1_real %.4f f1_fake %.4f auc %.4f',
                results['accuracy'], results['mac_f1'], results['f1_real'],
                results['f1_fake'], results['auc'])
    features = np.concatenate(feats) if feats else None
    feature_labels = np.concatenate(flabels) if flabels else None
    return results, labels, preds, ids, (features, feature_labels)
