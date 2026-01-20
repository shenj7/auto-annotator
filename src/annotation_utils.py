#!/usr/bin/env python3
import cv2
import numpy as np

def min_area_from_noise(noise: float) -> int:
    k = max(1, int(round(3 + 10 * noise)))
    return (k * k) * 3


def draw_local_contrast_dark_contours(
    image_path,
    noise=0.45,
    scale=5.0,
    contrast=0,
    min_area=None,
    max_area=None,
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

    # contours (external only) + area rejection
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if min_area is None:
        min_area = min_area_from_noise(noise)
    else:
        min_area = max(0, int(round(min_area)))
    if max_area is not None:
        max_area = max(0, int(round(max_area)))

    filtered = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered.append(c)
    cnts = filtered

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
    ap.add_argument("--min-area", type=float, default=None,
                    help="minimum blob area in pixels (default: derived from noise)")
    ap.add_argument("--max-area", type=float, default=None,
                    help="maximum blob area in pixels (default: no limit)")
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
        min_area=args.min_area,
        max_area=args.max_area,
        out_path=args.out,
        mask_out_path=args.mask_out,
    )
