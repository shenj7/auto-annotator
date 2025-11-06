#!/usr/bin/env python3
import cv2
import numpy as np

def simple_image_contours(image_path, low=50, high=150, blur=1.0, mode="external", thick=1, out_path="img_contours.png"):
    g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=blur, sigmaY=blur)

    edges = cv2.Canny(g, low, high)

    retrieval = cv2.RETR_EXTERNAL if mode == "external" else cv2.RETR_TREE
    cnts, _ = cv2.findContours(edges, retrieval, cv2.CHAIN_APPROX_SIMPLE)

    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), thick)
    cv2.imwrite(out_path, out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Make and draw contours of an image (Canny -> contours).")
    ap.add_argument("--image", help="path to input image")
    ap.add_argument("-l", "--low", type=float, default=50, help="Canny low threshold (default: 50)")
    ap.add_argument("-u", "--high", type=float, default=150, help="Canny high threshold (default: 150)")
    ap.add_argument("-b", "--blur", type=float, default=1.0, help="Gaussian sigma before Canny (default: 1.0)")
    ap.add_argument("-m", "--mode", default="external", choices=["external","tree"],
                    help="contour retrieval mode (default: external)")
    ap.add_argument("-t", "--thick", type=int, default=1, help="contour line thickness (default: 1)")
    ap.add_argument("-o", "--out", default="img_contours.png", help="output path (default: img_contours.png)")
    args = ap.parse_args()

    simple_image_contours(args.image, args.low, args.high, args.blur, args.mode, args.thick, args.out)
