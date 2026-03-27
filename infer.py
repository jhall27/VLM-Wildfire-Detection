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
    weights = torch.load(os.path.join(args.weight_dir, args.exp+'.pt'), map_location='cpu')
    model.load_state_dict(weights, strict=False)
    model.eval()
    model.to('cpu')

    image = Image.open(args.input_image)
    image = preprocess(image)
    # Model expects a batch, even if we only use one image.
    image = einops.rearrange(image, ' (a b) c d -> a b c d', a=1)

    outputs = model(inputs=image, plot_outputs=True)
    
    # Save a simple binary mask image.
    seg2 = torch.round(F.sigmoid(outputs[1][0, 0, :, :])).detach().cpu().numpy().astype(np.uint8) * 255
    output_name = os.path.splitext(os.path.basename(args.input_image))[0] + "_mask.png"
    Image.fromarray(seg2).save(os.path.join(args.output_dir, output_name))

if __name__ == '__main__':
    main()
