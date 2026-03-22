import sys
import os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import model
import importlib
import torch
import torch.optim as optim
import yaml
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse


def main(config):
    # ===================>> Setup <<===================
    dataloader = importlib.import_module("reader." + config.reader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cudnn.benchmark = False
    torch.cuda.empty_cache()

    data = config.data
    save = config.save
    params = config.params

    print("===> Read data <===")

    if data.isFolder:
        data, _ = ctools.readfolder(data)

    dataset = dataloader.loader(
        data,
        params.batch_size,
        shuffle=True,
        num_workers=4,
        sequence_length=2
    )

    print("===> Model building <===")
    net = model.Model()
    net.train()
    net.to(device)

    # Pretrain
    pretrain = config.pretrain
    if pretrain.enable:
        pretrain_path = pretrain.path
        print(f"===> Loading pretrained weights from {pretrain_path}")
        state = torch.load(pretrain_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            net.load_state_dict(state["model_state_dict"], strict=False)
        else:
            net.load_state_dict(state, strict=False)

    print("===> optimizer building <===")
    optimizer = optim.Adam(
        net.parameters(),
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.decay_step,
        gamma=params.decay
    )

    if params.warmup:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=params.warmup,
            after_scheduler=scheduler
        )

    savepath = os.path.join(save.metapath, save.folder, "checkpoint")
    os.makedirs(savepath, exist_ok=True)

    # ====================>> Training <<====================
    print("===> Training <===")

    length = len(dataset)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        torch.cuda.empty_cache()
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch + 1):
            for i, (data, label) in enumerate(dataset):
                # -------- move tensors to device --------
                for key in data:
                    if key != 'name':
                        data[key] = data[key].to(device)

                for key in label:
                    if torch.is_tensor(label[key]):
                        label[key] = label[key].to(device)

                # -------- forward --------
                total_loss, loss_dict = net.loss(
                    data['x_t'],
                    label['gaze'],
                    label['pose_prev'],
                    label['pose_curr']
                )

                # -------- backward --------
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                rest = timer.step() / 3600.0

                if i % 20 == 0:
                    log = (
                        f"[{epoch}/{params.epoch}] "
                        f"[{i}/{length}] "
                        f"total:{loss_dict['total_loss'].item():.6f} "
                        f"gaze:{loss_dict['gaze_loss'].item():.6f} "
                        f"pose:{loss_dict['pose_loss'].item():.6f} "
                        f"lr:{ctools.GetLR(optimizer)} "
                        f"rest time:{rest:.2f}h"
                    )
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

            scheduler.step()

            if epoch % save.step == 0:
                model_path = os.path.join(
                    savepath,
                    f"Iter_{epoch}_{save.model_name}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'iter': i,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                }, model_path)
                print(f"===> Model saved to {model_path}")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
    parser.add_argument('-s', '--train', type=str, required=True,
                        help='The source config for training.')
    parser.add_argument('-t', '--target', type=str,
                        help='config path about test')

    args = parser.parse_args()

    with open(args.train, 'r', encoding='utf-8') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))

    print("=====================>> (Begin) Training params << =======================")
    print(ctools.DictDumps(config))
    print("=====================>> (End) Training params << =======================")

    main(config.train)