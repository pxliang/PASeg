# pip install tifffile opencv-python scikit-image numpy
import numpy as np
import tifffile
import cv2
from skimage import morphology, filters
from skimage.io import imsave
import os

import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_img(img, mask, save_path=None):

    # ---- 绘图 ----
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def read_tiff_thumbnail(path, max_dim=4096):
    """
    只读 .tiff：返回 (thumb_rgb[H,W,3] uint8, scale)
    scale = base(最高分辨率)最长边 / 缩略图最长边
    """
    with tifffile.TiffFile(path) as tf:
        img = tf.series[0].asarray()
        # 计算缩放比例
        Ht, Wt = img.shape[:2]

        if max(Ht, Wt) > max_dim:
            scale = max(Ht, Wt) / float(max_dim)
        else:
            scale = 1.0
        img = cv2.resize(img, (int(Wt / scale), int(Ht / scale)), interpolation=cv2.INTER_AREA)

        return img, scale, Ht, Wt

def tissue_mask_lab(img_rgb, min_obj=5000, close_radius=3):
    """
    LAB 的 ab 能量 + Otsu，形态学清理，输出 uint8 {0,1}
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[...,1] - 128.0
    b = lab[...,2] - 128.0
    ab = np.sqrt(a*a + b*b)
    ab_u8 = cv2.normalize(ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    thr = filters.threshold_otsu(ab_u8)
    mask = (ab_u8 > thr)

    # 形态学：闭运算 → 填洞/去小块
    if close_radius > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_radius+1, 2*close_radius+1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)

    mask = morphology.remove_small_holes(mask, area_threshold=min_obj)
    mask = morphology.remove_small_objects(mask, min_size=min_obj)
    return mask.astype(np.uint8)

def get_foreground_mask_tiff(tiff_path, max_dim=4096, min_obj=5000, close_radius=3):
    """
    入口：只读 .tiff，返回 (mask, thumb, scale)
    - mask: 缩略图尺度的 uint8 {0,1}
    - thumb: 缩略图 RGB
    - scale: 从缩略图坐标映射回 base 尺度的比例
    """
    thumb, scale, H, W = read_tiff_thumbnail(tiff_path, max_dim=max_dim)
    mask = tissue_mask_lab(thumb, min_obj=min_obj, close_radius=close_radius)
    return mask, thumb, scale, H, W

def main(args):

    filenames = [f for f in os.listdir(args.tiff_path) if f.lower().endswith(('.tiff', '.tif'))]

    os.makedirs(args.output_plot_path, exist_ok=True)
    os.makedirs(args.output_mask_path, exist_ok=True)

    for fn in filenames:
        mask, thumb, scale, H, W = get_foreground_mask_tiff(os.path.join(args.tiff_path, fn), max_dim=4096, min_obj=8000, close_radius=3)

        print("thumb:", thumb.shape, "scale:", scale, "mask:", H, W)

        plot_img(thumb, mask, save_path=os.path.join(args.output_plot_path, f"{os.path.splitext(fn)[0]}.jpg"))

        mask = cv2.resize(mask.astype('uint8'), (W, H), interpolation=cv2.INTER_NEAREST)

        imsave(os.path.join(args.output_mask_path, f"{os.path.splitext(fn)[0]}.png"), mask)

# ========== 用法示例 ==========
if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--tiff_path", type=str, default='/data/TCGA-COAD/20x_images', help="Path to input .tiff (optional)")
    p.add_argument("--output_plot_path", type=str, default='./plots', help="Path to output overlay plots")
    p.add_argument("--output_mask_path", type=str, default='./masks', help="Path to output foreground masks")
    args = p.parse_args()

    main(args)
