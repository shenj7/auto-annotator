import os
import cv2
import glob
import numpy as np
from typing import Callable, List, Tuple
from pathlib import Path
from patch_utils import extract_patches

class ClassicalPipeline:
    """
    Classical CV pipeline to bootstrap the first dataset for active learning.
    Takes a directory of raw TEM images, runs an existing algorithm,
    and produces binary masks, contours, and an overlay image for oracle review.
    """
    def __init__(self, raw_data_dir: str, output_dir: str, algorithm_fn: Callable[[np.ndarray], np.ndarray]):
        """
        Args:
            raw_data_dir (str): Path to directory containing raw TEM images.
            output_dir (str): Path to save the extracted masks and overlays (e.g., pending_review).
            algorithm_fn (Callable): The existing algorithm function. It should take a grayscale
                                     image (numpy array) and return a binary mask (numpy array) 
                                     where 255 represents the foreground (dislocations) and 0 is background.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.algorithm_fn = algorithm_fn
        
        # Ensure output directory structure exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "overlays").mkdir(exist_ok=True)

    def process_all(self):
        """
        Process all images in the raw_data_dir using the algorithm_fn.
        """
        image_paths = sorted(glob.glob(str(self.raw_data_dir / "*.*")))
        # Filter basic image formats
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        print(f"Found {len(image_paths)} images to bootstrap.")

        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            self._process_single_image(img_path)

    def _process_single_image(self, img_path: Path):
        """
        Read one image, run the algorithmic pipeline, and save outputs as patches if needed.
        """
        # Read the image in grayscale for the algorithm
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Warning: Could not read image {img_path}")
            return

        stem = img_path.stem
        
        # We define a patch size for processing (e.g. 512x512)
        # If the image is smaller, extract_patches will just pad it to 512x512
        PATCH_SIZE = 128

        if img_gray.shape[0] <= PATCH_SIZE and img_gray.shape[1] <= PATCH_SIZE:
            # Small image, no need to patch logic, just wrap in list
            patches = [(img_gray, 0, 0)]
        else:
            # Large image, extract patches
            patches = extract_patches(img_gray, patch_size=(PATCH_SIZE, PATCH_SIZE), stride=(PATCH_SIZE, PATCH_SIZE))
            
            # Note: the pad function in extract_patches might pad the image to match stride. 
            # This is handled securely.

        total_contours = 0
        
        for patch_img, x, y in patches:
            # Also keep a BGR copy for the overlay
            img_bgr = cv2.cvtColor(patch_img, cv2.COLOR_GRAY2BGR)
    
            # 1. Run the existing algorithm to get the binary mask for the patch
            binary_mask = self.algorithm_fn(patch_img)
            
            # Ensure standard binary mask format (0 and 255)
            binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    
            # 2. Extract contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_contours += len(contours)
    
            # 3. Create overlay
            overlay_img = img_bgr.copy()
            cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)  # Red contours
    
            # Save the patch, mask, and overlay
            patch_name = f"{stem}_patch_x{x}_y{y}"
            
            cv2.imwrite(str(self.output_dir / "images" / f"{patch_name}.png"), patch_img)
            cv2.imwrite(str(self.output_dir / "masks" / f"{patch_name}.png"), binary_mask)
            cv2.imwrite(str(self.output_dir / "overlays" / f"{patch_name}_overlay.png"), overlay_img)
        
        print(f"Processed {img_path.name} into {len(patches)} patches: {total_contours} total boundaries found.")

# Example usage (for testing module standalone)
if __name__ == "__main__":
    def dummy_algorithm(img):
        # A simple thresholding dummy algorithm
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return thresh
        
    pipeline = ClassicalPipeline(
        raw_data_dir="raw_data", 
        output_dir="pending_review", 
        algorithm_fn=dummy_algorithm
    )
    # pipeline.process_all()
