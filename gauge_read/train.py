import os
import sys
import gc
import time
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from gauge_read.datasets import MeterDataset
from gauge_read.models.loss import TextLoss
from gauge_read.models.textnet import TextNet
from gauge_read.utils.augmentation import Augmentation
from gauge_read.utils.config import AttrDict
from gauge_read.utils.tools import AverageMeter, to_device, collate_fn
from gauge_read.utils.converter import StringLabelConverter

lr = None
train_step = 0
converter = StringLabelConverter()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gauge Read TextNet")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    return parser.parse_args()


def save_model(model, epoch, lr, optimzer, cfg):
    save_dir = os.path.join(cfg.experiment.save_dir, cfg.experiment.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "textgraph_{}_{}.pth".format(model.backbone_name, epoch))
    print("Saving to {}.".format(save_path))
    state_dict = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state_dict, save_path)


def train(model, train_loader, criterion, scheduler, optimizer, epoch, writer, cfg):
    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for i, (img, pointer_mask, dail_mask, text_mask, train_mask, transcripts, bboxs, mapping) in pbar:
        data_time.update(time.time() - end)

        train_step += 1

        img, pointer_mask, dail_mask, text_mask, train_mask = to_device(
            img, pointer_mask, dail_mask, text_mask, train_mask, device=cfg.system.device
        )

        output, pred_recog = model(img, bboxs, mapping)  # 4*12*640*640

        labels, label_lengths = converter.encode(transcripts.tolist())
        labels = to_device(labels, device=cfg.system.device)
        label_lengths = to_device(label_lengths, device=cfg.system.device)
        recog = (labels, label_lengths)

        loss_pointer, loss_dail, loss_text, loss_rec = criterion(
            output, pointer_mask, dail_mask, text_mask, train_mask, recog, pred_recog
        )

        loss = loss_pointer + loss_dail + loss_text + loss_rec

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        gc.collect()

        # Tensorboard logging
        if train_step == 1 or train_step % cfg.training.display_freq == 0:
            writer.add_scalar("Train/Loss", loss.item(), train_step)
            writer.add_scalar("Train/Pointer_Loss", loss_pointer.item(), train_step)
            writer.add_scalar("Train/Dial_Loss", loss_dail.item(), train_step)
            writer.add_scalar("Train/Text_Loss", loss_text.item(), train_step)
            writer.add_scalar("Train/Rec_Loss", loss_rec.item(), train_step)
            writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], train_step)
            writer.flush()

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Ptr": f"{loss_pointer.item():.4f}",
                "Dial": f"{loss_dail.item():.4f}",
                "Txt": f"{loss_text.item():.4f}",
                "Rec": f"{loss_rec.item():.4f}",
            }
        )

    if epoch % cfg.training.save_freq == 0:
        save_model(model, epoch, scheduler.get_last_lr()[0], optimizer, cfg)

    print("Training Loss: {}".format(losses.avg))


def main(cfg):
    global lr
    means = tuple(cfg.model.means)
    stds = tuple(cfg.model.stds)

    transform = Augmentation(size=640, mean=means, std=stds)

    trainset = MeterDataset(transform=transform, cfg=cfg)
    train_loader = data.DataLoader(
        trainset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.system.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    model = TextNet(backbone=cfg.model.net, is_training=True, cfg=cfg)

    model = model.to(cfg.system.device)

    if cfg.system.cuda:
        cudnn.benchmark = True

    criterion = TextLoss()

    lr = cfg.training.lr
    moment = cfg.training.momentum
    weight_decay = cfg.training.get("weight_decay", 1e-2)
    optim_name = str(cfg.training.get("optim", "AdamW")).lower()

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optim}. Use AdamW/Adam/SGD.")

    lr_adjust = str(cfg.training.get("lr_adjust", "cosine")).lower()
    if lr_adjust in {"cos", "cosine", "cosine_annealing"}:
        eta_min = cfg.training.get("eta_min", lr * 0.01)
        t_max = cfg.training.get("cosine_t_max", cfg.training.max_epoch + 1)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif lr_adjust == "step":
        step_size = cfg.training.get("step_size", 100)
        gamma = cfg.training.get("gamma", 0.90)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported lr_adjust: {cfg.training.lr_adjust}. Use cosine/step.")

    # Tensorboard writer
    log_dir = os.path.join("logs", cfg.experiment.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print("Start training")
    for epoch in range(cfg.training.start_epoch, cfg.training.start_epoch + cfg.training.max_epoch + 1):
        train(model, train_loader, criterion, scheduler, optimizer, epoch, writer, cfg)  # train
        scheduler.step()
    print("End.")

    writer.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)

    seed = int(cfg.training.get("seed", 114514))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg.print_config()
    main(cfg)
