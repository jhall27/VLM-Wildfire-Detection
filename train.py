import torch
from datasets.segmentation_data import WFSeg
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from utils.get_args import get_args
from eval import eval
from models.pidnet import get_seg_model
from pidnet_utils.configs import config
from pidnet_utils.criterion import BondaryLoss
from pidnet_utils.utils import FullModel
from utils.experiment import ensure_dirs, save_run_config, seed_everything
import einops


def write_logs(epoch, val_metrics, model, optim, args):
    # Pull out the validation numbers we want to keep after each epoch.
    val_loss = val_metrics['mean_val_loss']
    miou = val_metrics['mean_iou']
    precision = val_metrics.get('mean_precision', 0)
    recall = val_metrics.get('mean_recall', 0)
    f1 = val_metrics.get('mean_f1', 0)
    # Create a new log file if there isn't one
    try:
        if epoch == 0:
            df = pd.DataFrame(columns=['epoch', 'val_loss', 'mIoU', 'precision', 'recall', 'f1'])
        else:
            df = pd.read_csv(os.path.join(args.log_dir, args.exp+'.csv'))
    except:
        df = pd.DataFrame(columns=['epoch', 'val_loss', 'mIoU', 'precision', 'recall', 'f1'])

    # Save weights if the best loss is achieved
    if epoch == 0 or val_loss < np.amin(df['val_loss']):
        print('Best validation loss so far, saving weights.')
        save_weights(model, optim, args)
    if epoch == 0 or miou > np.amax(df['mIoU']):
        print('Best mIoU so far, saving weights.')
        save_weights(model, optim, args, miou=True)

    # Add new loss to df
    df.loc[len(df)] = [epoch, val_loss, miou, precision, recall, f1]
    # Save logs
    df.to_csv(os.path.join(args.log_dir, args.exp+'.csv'), index=False)


def save_weights(model, optim, args, miou=False):
    # Save a second copy when the best mIoU improves.
    if miou:
        torch.save(model.state_dict(), os.path.join(args.weight_dir, args.exp+'_miou.pt'))
        torch.save(optim.state_dict(), os.path.join(args.weight_dir, args.exp+'_optim_miou.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(args.weight_dir, args.exp+'.pt'))
        torch.save(optim.state_dict(), os.path.join(args.weight_dir, args.exp+'_optim.pt'))


def train(model, trainloader, valloader, args):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print('Starting training')
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        optim.zero_grad()
        total_batches = len(trainloader)
        progress = tqdm(
            enumerate(trainloader),
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
            disable=args.verbose,
        )
        last_batch_idx = -1
        for i, batch in progress:
            last_batch_idx = i
            # For quick smoke tests we can skip the train loop and just check validation.
            if args.test_val:
                break
            # Limit batches if we only want a short test run.
            if args.max_batches and i >= args.max_batches:
                break
            input = batch['img'].to(args.device)
            if not args.boxsup:
                mask = batch['mask'].to(args.device)
                boundary = batch['boundary'].to(args.device)
                # FullModel returns the training losses plus the segmentation outputs.
                losses, _, acc, loss_list = model(input, mask, boundary)
                loss = losses.mean()
            else:
                # Box supervision is kept separate from the normal mask-based baseline.
                loss = model(input, labels=None, bd_gt=None, id=None, 
                             box=batch['box'].to(args.device), 
                             lab_img=batch['lab_img'].to(args.device))
            
            last_loss = loss.item()
            epoch_loss += last_loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            progress.set_postfix(loss=f"{last_loss:.4f}")
        
            if i % 10 == 0 and args.verbose:
                print('Epoch: {:d}, batch: {:d}, Last training loss: {:.4f}'.format(epoch, i, last_loss))

        trained_batches = 0 if last_batch_idx < 0 or args.test_val else last_batch_idx + 1
        if args.max_batches:
            trained_batches = min(trained_batches, args.max_batches)
        mean_train_loss = 0.0 if trained_batches == 0 else (epoch_loss / trained_batches)
        print('Finished epoch: {:d}, training loss: {:.7f}. validating'.format(epoch, mean_train_loss))
        epoch_loss = 0
        # Validate
        loss_dict = eval(model, args, valloader)
        val_loss = loss_dict['mean_val_loss']
        miou = loss_dict['mean_iou']
        print('Finished validating, validation loss: {:.7f}'.format(val_loss))
        print('Mean index over union: {:.7f}'.format(miou))
        # Write logs and save weights
        write_logs(epoch, loss_dict, model, optim, args)


def main():
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, args.exp)
    # Make folders first so logs/checkpoints do not fail on a fresh clone.
    ensure_dirs([args.output_dir, args.log_dir, args.weight_dir])
    seed_everything(args.seed, deterministic=args.deterministic)
    
    # Keep the original teacher labels as the default, then swap in the
    # VLM-refined or fused labels only when we explicitly ask for them.
    if args.label_mode == 'vlm':
        args.sup_dir = 'vlm_masks'
    elif args.label_mode == 'fused':
        args.sup_dir = 'fused_masks'

    # Save the resolved config after label-mode routing so logs show the
    # actual supervision directory used for the run.
    save_run_config(args, os.path.join(args.log_dir, args.exp + "_config.json"))
    
    print(args)
    # Model size stays configurable so we can compare S / M / L later.
    config.MODEL.NAME = 'pidnet_'+args.model_size
    config.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_'+args.model_size.capitalize()+'_ImageNet.pth.tar'
    model = get_seg_model(cfg=config, imgnet_pretrained=True)
    # The smoke class is sparse, so use a positive class weight.
    pos_weight = einops.rearrange(torch.tensor([2.5]), '(a b c) -> a b c', a=1, b=1)
    sem_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model.to(args.device)
    
    # Train on the chosen label source (original SAM, VLM-filtered, or fused).
    trainset = WFSeg(root_dir=args.data_dir, 
                     mode='train',
                     boundary=(not args.boxsup),
                     include_id=args.include_id,
                     args=args)
    valset = WFSeg(root_dir=args.data_dir, 
                    mode='valid',
                    boundary=(not args.boxsup),
                     include_id=args.include_id,
                     args=args)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.debug_val:
        loss_dict = eval(model, args, valloader)
        print(loss_dict)
        exit()
    train(model, trainloader=trainloader, valloader=valloader, args=args)


if __name__ == "__main__":
    main()
