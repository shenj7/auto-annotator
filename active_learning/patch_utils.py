import numpy as np
from typing import List, Tuple, Dict
from PIL import Image

def extract_patches(image: np.ndarray, patch_size: Tuple[int, int] = (256, 256), stride: Tuple[int, int] = (256, 256)) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extracts small patches from a large image.
    If the image is not perfectly divisible by stride, it pads the bottom/right edges with zeros.
    
    Args:
        image (np.ndarray): The initial large image (H, W) or (H, W, C).
        patch_size (Tuple[int, int]): Size of patches as (H, W).
        stride (Tuple[int, int]): Stride size (H_stride, W_stride).
        
    Returns:
        List[Tuple[np.ndarray, int, int]]: A list of tuples containing:
            - The patch numpy array.
            - The x (column) coordinate.
            - The y (row) coordinate.
    """
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    ph, pw = patch_size
    sh, sw = stride

    # Calculate padding to ensure all patches are exactly patch_size
    pad_h = (sh - (h - ph) % sh) % sh
    pad_w = (sw - (w - pw) % sw) % sw

    if pad_h > 0 or pad_w > 0:
        if c == 1:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
        h, w = image.shape[:2]

    patches = []
    
    # We only want to yield patches that have at least *some* actual image data in them
    # Because we pad out to the full stride, the last patches might just have a tiny sliver of image
    for y in range(0, h - ph + 1, sh):
        for x in range(0, w - pw + 1, sw):
            if c == 1:
                patch = image[y:y+ph, x:x+pw]
            else:
                patch = image[y:y+ph, x:x+pw, :]
            patches.append((patch, x, y))

    return patches

def merge_patches(patches_with_coords: List[Tuple[np.ndarray, int, int]], original_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Stitches patches back into the original image shape. 
    Overrides overlapping areas (if stride < patch_size).
    Returns the exact original_shape, cropping any padding that was added.
    """
    if len(original_shape) == 2:
        h, w = original_shape
        c = 1
    else:
        h, w, c = original_shape

    # Find the bounds of the padded image
    max_h = max(y + patch.shape[0] for patch, x, y in patches_with_coords)
    max_w = max(x + patch.shape[1] for patch, x, y in patches_with_coords)

    if c == 1:
        merged = np.zeros((max_h, max_w), dtype=patches_with_coords[0][0].dtype)
    else:
        merged = np.zeros((max_h, max_w, c), dtype=patches_with_coords[0][0].dtype)

    for patch, x, y in patches_with_coords:
        ph, pw = patch.shape[:2]
        if c == 1:
            merged[y:y+ph, x:x+pw] = patch
        else:
            merged[y:y+ph, x:x+pw, :] = patch

    # Crop to original shape
    if c == 1:
        return merged[:h, :w]
    else:
        return merged[:h, :w, :]
