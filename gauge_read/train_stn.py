import os
import argparse
import random

import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from gauge_read.utils.tools import warp, warp_points, draw_points_on_batch
from gauge_read.datasets.meter_data import MeterSyn, STNTest
from gauge_read.models.stn import STNModel
from gauge_read.models.loss import STNLoss
from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger


def train(cfg):
    device = cfg.stn_training.device
    exp_name = f"stn_{cfg.experiment.exp_name}"
    save_dir = cfg.experiment.save_dir
    log_dir = os.path.join(save_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)
    logger.info("Starting STN training: exp_name=%s, device=%s", exp_name, device)

    trn_dataset = MeterSyn(
        size=cfg.stn_training.step * cfg.stn_training.batch_size,
        use_homography=(not cfg.stn_training.disable_homography),
        use_artefacts=(not cfg.stn_training.disable_artefacts),
        use_arguments=(not cfg.stn_training.disable_arguments),
    )
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=cfg.stn_training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    logger.info(
        "STN training dataset loaded: size=%s, batch_size=%s, num_workers=%s",
        len(trn_dataset),
        cfg.stn_training.batch_size,
        cfg.data.num_workers,
    )

    test_loader = None
    test_dir = cfg.data.get("stn_test_path")
    if test_dir and os.path.exists(test_dir):
        test_dataset = STNTest(test_dir)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.stn_training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        logger.info("STN validation dataset loaded from %s, size=%s", test_dir, len(test_dataset))
    elif test_dir:
        logger.warning("STN validation directory not found: %s", test_dir)

    model_stn = STNModel(pretrained=True).to(device)
    optimizer = optim.AdamW(model_stn.parameters(), lr=cfg.stn_training.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.stn_training.epochs * len(trn_loader)), eta_min=cfg.stn_training.lr_min
    )
    criterion = STNLoss()
    logger.info("STN model, optimizer, scheduler, and criterion initialized")

    total_loss = 0.0
    best_loss = float("inf")
    for ep in range(cfg.stn_training.epochs):
        logger.info("Starting STN epoch %s/%s", ep + 1, cfg.stn_training.epochs)
        train_pbar = tqdm(trn_loader, total=len(trn_loader), desc=f"Epoch {ep}")
        for i, (img, Minv, center) in enumerate(train_pbar):
            model_stn.train()
            optimizer.zero_grad()

            img = img.float().to(device)
            Minv = Minv.to(device)
            center = center.to(device)

            Minv_pred, pred_st, pred_center = model_stn(img)

            loss, loss_reg, loss_center = criterion(pred_st, Minv, pred_center, center)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if i == 0:
                logger.debug(
                    "STN epoch %s first batch shapes: img=%s, Minv=%s, center=%s",
                    ep,
                    tuple(img.shape),
                    tuple(Minv.shape),
                    tuple(center.shape),
                )

            global_step = ep * len(trn_loader) + i
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("STN/lr", current_lr, global_step)
            writer.add_scalar("STN/loss", loss.item(), global_step)
            writer.add_scalar("STN/loss_reg", loss_reg.item(), global_step)
            writer.add_scalar("STN/loss_center", loss_center.item(), global_step)

            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "l_reg": f"{loss_reg.item():.4f}",
                    "l_cen": f"{loss_center.item():.4f}",
                    "avg": f"{(total_loss / (global_step + 1)):.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            if global_step == 0 or global_step % 100 == 0:
                logger.debug(
                    "STN train step %s summary: loss=%.4f reg=%.4f center=%.4f lr=%.6e",
                    global_step,
                    loss.item(),
                    loss_reg.item(),
                    loss_center.item(),
                    current_lr,
                )

            if i == 0:
                points_pixel = center * 224.0
                warped_points = warp_points(points_pixel, Minv_pred, device=device, sz=224)
                img_with_pt = draw_points_on_batch(img, points_pixel, color=(1.0, 0.0, 0.0))

                img_warped = warp(img, Minv_pred, device=device)
                img_warped_with_pt = draw_points_on_batch(img_warped, warped_points, color=(1.0, 0.0, 0.0))

                writer.add_images("STN/train_original", img_with_pt, global_step)
                writer.add_images("STN/train_warped", img_warped_with_pt, global_step)

        if test_loader:
            model_stn.eval()
            with torch.no_grad():
                logger.info("Running STN validation for epoch %s", ep)
                test_pbar = tqdm(test_loader, total=len(test_loader), desc=f"Epoch {ep}", leave=False)
                for i, img in enumerate(test_pbar):
                    img = img.to(device)
                    Minv_pred, _, pred_center = model_stn(img)

                    if i == 0:
                        points_pixel = pred_center * 224.0
                        warped_points = warp_points(points_pixel, Minv_pred, device=device, sz=224)

                        img_with_pt = draw_points_on_batch(img, points_pixel, color=(1.0, 0.0, 0.0))

                        img_warped = warp(img, Minv_pred, device=device)
                        img_warped_with_pt = draw_points_on_batch(img_warped, warped_points, color=(1.0, 0.0, 0.0))

                        writer.add_images("STN/test_original", img_with_pt, global_step)
                        writer.add_images("STN/test_warped", img_warped_with_pt, global_step)

        ep_loss = total_loss / ((ep + 1) * len(trn_loader))
        logger.info("Finished STN epoch %s: avg_loss=%.6f, best_loss=%.6f", ep, ep_loss, best_loss)

        if ep_loss < best_loss:
            best_loss = ep_loss
            logger.info("New best loss %.4f achieved at epoch %s, saving model checkpoint", best_loss, ep)
            save_path = os.path.join(save_dir, exp_name, f"stn_ep{ep}_loss{ep_loss:.4f}.pth")
            torch.save(model_stn.state_dict(), save_path)

    writer.close()
    logger.info("STN training finished: best_loss=%.6f", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STN model")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.debug:
        import logging
        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("train_stn console log level set to DEBUG")

    cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    logger.info("Launching train_stn.py with config=%s", args.config or AttrDict.DEFAULT_CONFIG_PATH)

    seed = int(cfg.stn_training.get("seed", cfg.training.get("seed", 114514)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("STN training random seeds initialized to %s", seed)

    cfg.print_config()
    train(cfg)
