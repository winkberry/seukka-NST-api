"""
Minimum inference code
"""
import os
import numpy as np
from imageio import imwrite
from PIL import Image
import tensorflow as tf



def main(m_path, img_path, out_dir):
    imported = tf.saved_model.load(m_path)
    f = imported.signatures["serving_default"]
    img = np.array(Image.open(img_path).convert("RGB"))
    img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
    out = f(tf.constant(img))['output_1']
    print(out)
    out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if out_dir == "":
        out_dir = "."
    out_path = os.path.join(out_dir, os.path.split(img_path)[1])
    imwrite(out_path, out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str,
                        default=os.path.join("exported_models", "light_paprika_SavedModel"))
    parser.add_argument("--img_path", type=str,
                        default=os.path.join("input_images", "temple.jpeg"))
    parser.add_argument("--out_dir", type=str, default='out')
    args = parser.parse_args()
    main(args.m_path, args.img_path, args.out_dir)
