#!/usr/bin/env python3
import cv2
import numpy as np

def draw_local_contrast_dark_contours(
    image_path,
    noise=0.45,
    scale=5.0,
    contrast=0,
    out_path="contours.png",
    mask_out_path=None,
):
    g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # optional pre-smoothing tied to noise to quiet sensor speckle
    sigma_pre = 1.0 + 2.5 * noise
    g0 = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma_pre, sigmaY=sigma_pre)

    # local neighborhood mean (bigger 'scale' -> larger context)
    local = cv2.GaussianBlur(g0, (0, 0), sigmaX=scale, sigmaY=scale)

    # local contrast: positive where pixel is darker than its neighbors
    diff = cv2.subtract(local, g0)  # saturates at 0..255

    # threshold on local contrast
    _, bw = cv2.threshold(diff, int(contrast), 255, cv2.THRESH_BINARY)

    # morphology cleanup scales with noise
    k = max(1, int(round(3 + 10 * noise)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # contours (external only) + tiny-area rejection tied to kernel size
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (k * k) * 3
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

    # draw and save
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 1)
    cv2.imwrite(out_path, out)

    if mask_out_path:
        mask = np.full_like(g, 255)
        if cnts:
            cv2.drawContours(mask, cnts, -1, 0, thickness=cv2.FILLED)
        cv2.imwrite(mask_out_path, mask)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Outline dark spots using local-contrast thresholding.")
    ap.add_argument("--image", help="path to input image")
    ap.add_argument("-s", "--sensitivity", type=float, default=0.45,
                    help="0..1; higher = ignore more small noise (default: 0.45)")
    ap.add_argument("-S", "--scale", type=float, default=5.0,
                    help="Gaussian sigma for neighborhood size in pixels (default: 5.0)")
    ap.add_argument("-c", "--contrast", type=float, default=0,
                    help="local contrast threshold (local_mean - pixel) in 0..255 (default: 0)")
    ap.add_argument(
        "-o",
        "--out",
        default="contours.png",
        help="output path (default: contours.png)",
    )
    ap.add_argument(
        "-m",
        "--mask-out",
        default=None,
        help="optional path to write binary mask (white background, black foreground)",
    )
    args = ap.parse_args()

    draw_local_contrast_dark_contours(
        args.image,
        noise=args.sensitivity,
        scale=args.scale,
        contrast=args.contrast,
        out_path=args.out,
        mask_out_path=args.mask_out,
    )
