import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from gauge_read.utils.tools import warp, warp_points, draw_points_on_batch
from gauge_read.datasets.meter_data import MeterSyn, STNTest
from gauge_read.models.stn import STNModel
from gauge_read.models.loss import STNLoss


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(logdir=f"logs/{args.exp_name}")

    trn_dataset = MeterSyn(
        size=args.step * args.batch_size,
        use_homography=(not args.disable_homography),
        use_artefacts=(not args.disable_artefacts),
        use_arguments=(not args.disable_arguments),
    )
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Training dataset loaded, size: {len(trn_dataset)}")

    test_loader = None
    if args.test_dir and os.path.exists(args.test_dir):
        test_dataset = STNTest(args.test_dir)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Validation dataset loaded from {args.test_dir}, size: {len(test_dataset)}")

    model_stn = STNModel(pretrained=True).to(device)
    optimizer = optim.AdamW(model_stn.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(trn_loader)), eta_min=args.lr_min)
    criterion = STNLoss()

    if args.resume_path:
        model_stn.load_state_dict(torch.load(args.resume_path))

    total_loss = 0.0
    best_loss = float("inf")
    for ep in range(args.epochs):
        train_pbar = tqdm(trn_loader, total=len(trn_loader), desc=f"Train Epoch {ep}", leave=False)
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

            global_step = ep * len(trn_loader) + i
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/lr", current_lr, global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/loss_reg", loss_reg.item(), global_step)
            writer.add_scalar("train/loss_center", loss_center.item(), global_step)
            writer.add_scalar("train/avg_loss", total_loss / (global_step + 1), global_step)

            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "l_reg": f"{loss_reg.item():.4f}",
                    "l_cen": f"{loss_center.item():.4f}",
                    "avg": f"{(total_loss / (global_step + 1)):.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            if i == 0:
                # 转换坐标到像素坐标 (sz=224)
                points_pixel = center * 224.0
                # 对点进行 warp
                warped_points = warp_points(points_pixel, Minv_pred, device=device, sz=224)

                # 在原图和变换图上画红点
                img_with_pt = draw_points_on_batch(img, points_pixel, color=(1.0, 0.0, 0.0))

                img_warped = warp(img, Minv_pred, device=device)
                img_warped_with_pt = draw_points_on_batch(img_warped, warped_points, color=(1.0, 0.0, 0.0))

                writer.add_images("train/original", img_with_pt, global_step)
                writer.add_images("train/warped", img_warped_with_pt, global_step)

        if test_loader:
            model_stn.eval()
            with torch.no_grad():
                test_pbar = tqdm(test_loader, total=len(test_loader), desc=f"Test Epoch {ep}", leave=False)
                for i, img in enumerate(test_pbar):
                    img = img.to(device)
                    Minv_pred, _, pred_center = model_stn(img)

                    if i == 0:
                        points_pixel = pred_center * 224.0
                        warped_points = warp_points(points_pixel, Minv_pred, device=device, sz=224)

                        img_with_pt = draw_points_on_batch(img, points_pixel, color=(1.0, 0.0, 0.0))

                        img_warped = warp(img, Minv_pred, device=device)
                        img_warped_with_pt = draw_points_on_batch(img_warped, warped_points, color=(1.0, 0.0, 0.0))

                        writer.add_images("test/original", img_with_pt, global_step)
                        writer.add_images("test/warped", img_warped_with_pt, global_step)

        ep_loss = total_loss / ((ep + 1) * len(trn_loader))
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(model_stn.state_dict(), f"logs/{args.exp_name}/ep{ep}_loss{ep_loss:.4f}.pth")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="stn_convnext")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--disable_homography", action="store_true")
    parser.add_argument("--disable_artefacts", action="store_true")
    parser.add_argument("--disable_arguments", action="store_true")
    args = parser.parse_args()
    train(args)
