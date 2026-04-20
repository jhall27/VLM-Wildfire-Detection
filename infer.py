from PIL import Image
import os
import torch
import einops
import numpy as np

from utils.plot_outputs import *
from utils.get_args import get_args
from utils.experiment import ensure_dirs, seed_everything

from pidnet_utils.configs import config
from models.pidnet import get_seg_model
import torch.nn.functional as F
from pidnet_utils.criterion import BondaryLoss
from pidnet_utils.utils import FullModel
from torchvision.transforms import v2


def preprocess(img):
    # Match the same basic image preprocessing used by training/eval.
    img = v2.functional.to_image(img)
    img = v2.functional.to_dtype(img, torch.uint8)
    img = v2.functional.resize(img, [1080, 1920], antialias=True)
    img = v2.functional.to_dtype(img, torch.float32, scale=True)
    img = v2.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img


def main():
    args = get_args()
    ensure_dirs([args.output_dir])
    seed_everything(args.seed, deterministic=args.deterministic)
    cfg = config
    cfg.MODEL.NAME = 'pidnet_'+args.model_size
    cfg.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_'+args.model_size.capitalize()+'_ImageNet.pth.tar'
    model = get_seg_model(cfg=cfg, imgnet_pretrained=True)
    pos_weight = einops.rearrange(torch.tensor([2.5]), '(a b c) -> a b c', a=1, b=1)
    sem_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bd_criterion = BondaryLoss()

    
    model = FullModel(model, sem_criterion, bd_criterion)
    print('Loading model weights')
    weight_path = os.path.join(args.weight_dir, args.exp+'.pt')
    try:
        weights = torch.load(weight_path, map_location='cpu', weights_only=True)
    except TypeError:
        # Older torch versions do not support weights_only yet.
        weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(weights, strict=False)
    model.eval()
    model.to(args.device)

    image = Image.open(args.input_image).convert('RGB')
    vis_image = image.resize((1920, 1080))
    image = preprocess(image).to(args.device)
    # Model expects a batch, even if we only use one image.
    image = einops.rearrange(image, ' (a b) c d -> a b c d', a=1)

    with torch.no_grad():
        outputs = model(inputs=image, plot_outputs=True)

    prob = F.sigmoid(outputs[1][0, 0, :, :]).detach().cpu().numpy()
    strict_mask = (prob >= 0.5).astype(np.uint8) * 255
    loose_mask = (prob >= 0.2).astype(np.uint8) * 255
    prob_img = (prob * 255).clip(0, 255).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    Image.fromarray(strict_mask).save(os.path.join(args.output_dir, base_name + "_mask.png"))
    Image.fromarray(loose_mask).save(os.path.join(args.output_dir, base_name + "_mask_loose.png"))
    Image.fromarray(prob_img).save(os.path.join(args.output_dir, base_name + "_prob.png"))

    # Create a simple red overlay so weak smoke predictions remain visible.
    vis_arr = np.array(vis_image).astype(np.float32)
    overlay = vis_arr.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + prob * 180.0, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.35 * prob), 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - 0.35 * prob), 0, 255)
    Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(args.output_dir, base_name + "_overlay.png"))

if __name__ == '__main__':
    main()
