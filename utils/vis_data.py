import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import einops
import os
from torchvision.transforms.functional import resize
import plot_outputs
from PIL import Image
from input_utils import read_txt_box


def center_crop(im, size):
    width, height = im.size   # Get dimensions
    new_width, new_height = size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def main(img_paths=["/home/julius/wf_rt_seg_public/data/images/test/karkkila_DJI_0005_frame160.jpg"], vis_manual=False, output_dir='data_vis'):
    fig, ax = plt.subplots(2, 3, figsize=(30, 11.5))
    for i, img_path in enumerate(img_paths):
        if vis_manual:
            mask_path = img_path.replace('images', 'manual_masks').split('.')[0] + '.png'
        else:
            mask_path = img_path.replace('images', 'sam_masks').split('.')[0] + '.png'
        box_path = img_path.replace('images', 'labels').split('.')[0] + '.txt'
        img = Image.open(img_path)
        img = center_crop(img, (1920, 1080))
        mask = Image.open(mask_path)
        mask = center_crop(mask, (1920, 1080))
        box = read_txt_box(box_path, einops.rearrange(np.array(img), 'h w c -> w h c'))

        ax[i//3, i%3].imshow(img)
        plot_outputs.show_box(box, ax[i//3, i%3])
        plot_outputs.show_mask(np.array(mask)/255, ax[i//3, i%3])
        ax[i//3, i%3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_imgs.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    main(img_paths=["data/images/test/karkkila_DJI_0005_frame160.jpg",
                    "data/images/test/evoDJI_0009_frame113.jpg",
                    "data/images/test/ruokolahti_DJI_0086_frame152.jpg",
                    "data/images/test/heinola_DJI_0028_frame170.jpg",
                    "data/images/test/karkkila_DJI_0008_frame75.jpg",
                    "data/images/test/evoDJI_0001_frame136.jpg"], 
                    vis_manual=False, output_dir='data_vis')
