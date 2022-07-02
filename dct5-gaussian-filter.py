import argparse
import cv2
import glob
import math
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm


EXT = "*.bmp"

SIGMA = 2
RADIUS = round(SIGMA * 3.3)
DIM = 4


def get_kernel(sigma, radius, dim, phi):
    kernel = [0.0] * dim
    half_sq_sphi = 0.5 * sigma * sigma * phi * phi

    for k in range(dim):
        if k == 0:
            kernel[k] = math.exp(-half_sq_sphi * k * k) / ((2 * radius) + 1)
        else:
            kernel[k] = math.exp(-half_sq_sphi * k * k) / (radius + 0.5)

    return kernel


def gaussian_filter(image, sigma, radius, dim):
    height, width = image.shape[:2]
    col_end = width - 1

    fk = [0.0] * width
    phi = math.pi / (radius + 0.5)
    gk = get_kernel(sigma, radius, dim, phi)

    image_1d = np.ravel(image.tolist())
    tmp_1d = [0.0] * len(image_1d)
    for y in range(height):
        skip = width * y

        for k in range(dim):
            c1 = 2.0 * math.cos(phi * k)
            cr = math.cos(phi * k * radius)

            fk[0] = 0.0
            fk[1] = 0.0
            for u in range(-radius, (radius + 1)):
                fk[0] += image_1d[skip + abs(u)] * math.cos(phi * k * u)
                fk[1] += image_1d[skip + abs(u + 1)] * math.cos(phi * k * u)

            tmp_1d[skip] += fk[0] * gk[k]
            tmp_1d[skip + 1] += fk[1] * gk[k]

            for x in range(1, col_end):
                delta = image_1d[skip + (col_end - abs(col_end - (x + radius + 1)))] \
                        - image_1d[skip + (col_end - abs(col_end - (x + radius)))] \
                        - image_1d[skip + abs(x - radius)] \
                        + image_1d[skip + abs(x - radius - 1)]

                fk[x + 1] = (c1 * fk[x]) - fk[x - 1] + (cr * delta)
                tmp_1d[skip + x + 1] += fk[x + 1] * gk[k]

    return np.array(tmp_1d).reshape(height, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help = "input directory")
    parser.add_argument("output_dir", help = "output directory")
    parser.add_argument("--scale", default = "1.0", help = "scale of image")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    scale = args.scale

    if(os.path.exists(output_dir)):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    files = glob.glob(input_dir + "/" + EXT)


    for file in tqdm(files):
        file_name = os.path.basename(file)
        
        src = cv2.imread(file)
        src = cv2.resize(src, dsize = None, fx = float(scale), fy = float(scale))
        h, w, ch = src.shape[:3]

        dst = np.empty((h, w, ch))

        for idx in range(ch):
            # horizontal
            blr = gaussian_filter(src[:, :, idx], SIGMA, RADIUS, DIM)

            # vertical
            blr = gaussian_filter(blr.T, SIGMA, RADIUS, DIM)

            dst[:, :, idx] = blr.T

        cv2.imwrite((output_dir + "/" + file_name), dst)


main()