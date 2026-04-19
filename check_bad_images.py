import os
from PIL import Image

img_dir = "data/images/train"  # change to train folder
bad_images = []

for root, _, files in os.walk(img_dir):
    for f in files:
        path = os.path.join(root, f)
        try:
            with Image.open(path) as im:
                im.verify()  # verify will raise an exception if broken
        except Exception as e:
            print(f"Bad image: {path} -> {e}")
            bad_images.append(path)

print(f"Total bad images: {len(bad_images)}")