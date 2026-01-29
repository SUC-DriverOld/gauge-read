import os
import gc
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataset import Meter
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from util.config import config as cfg, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.shedule import FixLR
from util.tool import collate_fn
from util.converter import StringLabelConverter

lr = None
train_step = 0
converter = StringLabelConverter()


def save_model(model, epoch, lr, optimzer):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, "textgraph_{}_{}.pth".format(model.backbone_name, epoch))
    print("Saving to {}.".format(save_path))
    state_dict = {
        "lr": lr,
        "epoch": epoch,
        "model": model.state_dict() if not cfg.mgpu else model.state_dict(),
        "optimizer": optimzer.state_dict(),
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print("Loading from {}".format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])


def train(model, train_loader, criterion, scheduler, optimizer, epoch, writer):
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

        img, pointer_mask, dail_mask, text_mask, train_mask = to_device(img, pointer_mask, dail_mask, text_mask, train_mask)

        output, pred_recog = model(img, bboxs, mapping)  # 4*12*640*640

        labels, label_lengths = converter.encode(transcripts.tolist())
        labels = to_device(labels)
        label_lengths = to_device(label_lengths)
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
        if train_step == 1 or train_step % cfg.display_freq == 0:
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

    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_last_lr()[0], optimizer)

    print("Training Loss: {}".format(losses.avg))


def main():
    global lr
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds)

    trainset = Meter(transform=transform)
    train_loader = data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Model
    model = TextNet(backbone=cfg.net, is_training=True)

    model = model.to(cfg.device)

    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)

    criterion = TextLoss()

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == "Synthtext":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == "Synthtext":
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

    # Tensorboard writer
    log_dir = os.path.join("logs", cfg.exp_name)
    if not os.path.exists(log_dir):
        mkdirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("Start training TextGraph_welcomeMEddpnew::--")
    for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.max_epoch + 1):
        train(model, train_loader, criterion, scheduler, optimizer, epoch, writer)  # train
        scheduler.step()
    print("End.")

    writer.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    print_config(cfg)

    # main
    main()
