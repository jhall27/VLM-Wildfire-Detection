import torch
from utils.get_args import get_args
from utils.plot_outputs import plot_outputs
import numpy as np
from datasets.segmentation_data import WFSeg
import os
from torch.utils.data import DataLoader
from pidnet_utils.configs import config
import torch.nn.functional as F
from models.pidnet import get_seg_model
from pidnet_utils.criterion import BondaryLoss
from pidnet_utils.utils import FullModel
import einops
from torchmetrics.classification import BinaryJaccardIndex
from itertools import cycle
from torcheval.metrics.functional import binary_precision, binary_recall, binary_accuracy, binary_f1_score
import pandas as pd
from utils.experiment import ensure_dirs, get_metric_template, measure_inference_speed, save_run_config, seed_everything


def eval(model, args, loader):
    model.to(args.device)
    model.eval()
    total_loss = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_rand = 0
    total_f1 = 0
    with torch.no_grad():
        if args.vis_val:
            try:
                os.mkdir(args.output_dir)
            except:
                if not args.force:
                    print('Warning: eval dir already exists.')
                    print('Overwrite with argument --force or -f')
                    exit()
            print('Saving output images to dir '+args.output_dir)
        for i, batch in enumerate(loader):
            # Useful for fast checks when we do not want to run the whole split.
            if args.max_batches and i >= args.max_batches:
                break
            input = batch['img'].to(args.device)
            mask = batch['mask'].to(args.device)
            boundary = batch['boundary'].to(args.device)
            if args.include_id:
                _, outputs, acc, loss_list = model(input, mask, boundary, id=batch['id'])
            else:
                _, outputs, acc, loss_list = model(input, mask, boundary)
            loss = loss_list[0]
            acc = acc.mean()
            total_iou += acc.item()

            binary_mask_pred = torch.flatten(torch.round(F.sigmoid(outputs[1])))
            binary_gt = torch.flatten(mask)
            prec = precision(binary_mask_pred, binary_gt).item()
            total_precision += prec
            # Recall was more stable here after casting to uint8.
            rec = recall(binary_mask_pred.to(torch.uint8), binary_gt.to(torch.uint8)).item()
            total_recall += rec
            rand = rand_ind(binary_mask_pred, binary_gt).item()
            total_rand += rand
            f1 = dice(binary_mask_pred, binary_gt).item()
            total_f1 += f1
            
            # Generate outputs
            if args.vis_val:
                plot_outputs(i, batch, F.sigmoid(outputs[1]), args, re_s=False)

            total_loss += loss.item()
    
    loss_dict = {
        "model":args.exp,
        "mean_val_loss":total_loss/(i+1),
        "mean_iou":total_iou/(i+1),
        "mean_rand":total_rand/(i+1),
        "mean_precision":total_precision/(i+1),
        "mean_recall":total_recall/(i+1),
        "mean_f1":total_f1/(i+1)
    }
    return loss_dict


def eval_sam(args, sam_loader, gt_loader):
    iou = BinaryJaccardIndex(zero_division=1.0e-9)
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_rand = 0
    total_f1 = 0
    for i, (sam_batch, gt_batch) in enumerate(zip(cycle(sam_loader), gt_loader)):
        # Same batch limiter as the student eval path.
        if args.max_batches and i >= args.max_batches:
            break
        sam_mask = sam_batch['mask']
        gt_mask = gt_batch['mask']
        total_iou += iou(sam_mask, gt_mask).item()

        binary_mask_pred = torch.flatten(sam_mask)
        binary_gt = torch.flatten(gt_mask)
        prec = precision(binary_mask_pred, binary_gt).item()
        total_precision += prec
        rec = recall(binary_mask_pred.to(torch.uint8), binary_gt.to(torch.uint8)).item()
        total_recall += rec
        rand = rand_ind(binary_mask_pred, binary_gt).item()
        total_rand += rand
        f1 = dice(binary_mask_pred, binary_gt).item()
        total_f1 += f1
    
    loss_dict = {
        "model":args.exp,
        "mean_val_loss":0,
        "mean_iou":total_iou/(i+1),
        "mean_rand":total_rand/(i+1),
        "mean_precision":total_precision/(i+1),
        "mean_recall":total_recall/(i+1),
        "mean_f1":total_f1/(i+1)
    }
    
    return loss_dict


def precision(preds, gt):
    return binary_precision(preds, gt)


def recall(preds, gt):
    return binary_recall(preds, gt)


def rand_ind(preds, gt):
    return binary_accuracy(preds, gt)


def dice(preds, gt):
    return binary_f1_score(preds, gt)


def eval_teacher(manual_loader, teacher_dir, args):
    # Compare teacher masks directly against the manual masks.
    args.sup_dir = teacher_dir
    args.manual_masks = False
    args.eval_snake = True
    teacher_mask_set = WFSeg(args.data_dir, args.mode, manual_masks=False, boundary=True, args=args)
    teacher_loader = DataLoader(teacher_mask_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    loss_dict = eval_sam(args, teacher_loader, manual_loader)
    return loss_dict


def main():
    args = get_args()
    ensure_dirs([args.output_dir, args.log_dir, args.weight_dir])
    seed_everything(args.seed, deterministic=args.deterministic)
    save_run_config(args, os.path.join(args.log_dir, "eval_config.json"))
    print(args)
    # Eval is simpler to manage one image at a time.
    args.batch_size = 1
    loss_list = []

    wf_set = WFSeg(args.data_dir, args.mode, manual_masks=True, boundary=True, args=args)

    sets = [(wf_set,'AI For Mankind data')]

    if len(args.single_model) == 0:
        teachers = [
            'sam_masks'
        ]
    else:
        teachers = []

    for set, name in sets[:1]:
        eval_loader = DataLoader(set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for teacher_dir in teachers:
            args.exp = teacher_dir
            losses = {'dataset':name}
            losses.update(eval_teacher(eval_loader, teacher_dir, args))
            print(teacher_dir)
            print(losses)
            loss_list.append(losses)
    
    # Tuple: (exp_name, pidnet_size)
    if args.eval_sam:
        students = []
    elif len(args.single_model) != 0:
        students = [(args.single_model.split('/')[-1][:-3], args.model_size)]
    else:
        students = [
            ('sam_sup_pidnet_s', 's'),
            ('sam_sup_pidnet_m', 'm'),
            ('sam_sup_pidnet_l', 'l'),
            ('loss_ablation_pidnet_s', 's'),
        ]

    
    for student_model, pidnet_size in students:
        if student_model != 'sam_sup_pidnet_s':
            eval_sets = sets[:1]
        else:  
            eval_sets = sets
        for set, name in eval_sets:
            eval_loader = DataLoader(set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            args.exp = student_model
            config.MODEL.NAME = 'pidnet_'+pidnet_size
            config.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_'+pidnet_size.capitalize()+'_ImageNet.pth.tar'
            model = get_seg_model(cfg=config, imgnet_pretrained=True)
            # Keep eval consistent with the student model setup.
            pos_weight = einops.rearrange(torch.tensor([1]), '(a b c) -> a b c', a=1, b=1)
            sem_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            bd_criterion = BondaryLoss()
            model = FullModel(model, sem_criterion, bd_criterion)
            model.to(args.device)
            print('Loading model weights')
            weights = torch.load(os.path.join(args.weight_dir, args.exp+'.pt'), map_location='cpu')
            model.load_state_dict(weights, strict=False)
            
            losses = {'dataset':name}
            losses.update(eval(model, args, eval_loader))
            # Use one batch from the eval loader to get a quick speed estimate.
            speed_metrics = measure_inference_speed(
                model,
                next(iter(eval_loader)),
                args.device,
                warmup=args.speed_warmup,
                steps=args.speed_steps,
            )
            losses.update(speed_metrics)

            print(student_model)
            print(losses)
            losses['model'] = student_model
            loss_list.append(losses)

    df = pd.DataFrame.from_dict(loss_list) 
    df.to_csv(args.metrics_output, float_format='%.3f', index=False)

    template_rows = []
    template_row = get_metric_template()
    for losses in loss_list:
        template = template_row.copy()
        template.update({
            "experiment": losses.get("model", ""),
            "dataset_split": args.mode,
            "teacher_model": "SAM",
            "student_model": losses.get("model", ""),
            "mIoU": f"{losses.get('mean_iou', ''):.3f}" if isinstance(losses.get("mean_iou"), float) else losses.get("mean_iou", ""),
            "precision": f"{losses.get('mean_precision', ''):.3f}" if isinstance(losses.get("mean_precision"), float) else losses.get("mean_precision", ""),
            "recall": f"{losses.get('mean_recall', ''):.3f}" if isinstance(losses.get("mean_recall"), float) else losses.get("mean_recall", ""),
            "f1": f"{losses.get('mean_f1', ''):.3f}" if isinstance(losses.get("mean_f1"), float) else losses.get("mean_f1", ""),
            "milliseconds_per_image": f"{losses.get('milliseconds_per_image', ''):.3f}" if isinstance(losses.get("milliseconds_per_image"), float) else losses.get("milliseconds_per_image", ""),
            "fps": f"{losses.get('fps', ''):.3f}" if isinstance(losses.get("fps"), float) else losses.get("fps", ""),
            "checkpoint": args.single_model or os.path.join(args.weight_dir, losses.get("model", "") + ".pt"),
        })
        template_rows.append(template)
    pd.DataFrame(template_rows).to_csv(args.results_template, index=False)

if __name__ == "__main__":
    main()
