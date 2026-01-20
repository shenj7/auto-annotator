import argparse
from pathlib import Path
import csv
import math
import cv2
from video_utils import extract_frames
from gui import main as gui_main
from annotation_utils import draw_local_contrast_dark_contours
MAX_FRAMES = 100


def get_last_parameters_from_csv(settings_file):
    """Read the last row from settings.csv to get the most recent parameters"""
    if not settings_file.exists():
        raise ValueError(f"Settings file not found: {settings_file}")
    
    with open(settings_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if not rows:
            raise ValueError("No parameters found in settings.csv")
        
        # Get the last row (most recent)
        last_row = rows[-1]
        
        scale_percent = None
        if 'scale_percent' in last_row and last_row['scale_percent'] != '':
            try:
                scale_percent = float(last_row['scale_percent'])
            except ValueError:
                scale_percent = None

        min_radius = None
        if 'min_radius' in last_row and last_row['min_radius'] != '':
            try:
                min_radius = float(last_row['min_radius'])
            except ValueError:
                min_radius = None

        max_radius = None
        if 'max_radius' in last_row and last_row['max_radius'] != '':
            try:
                max_radius = float(last_row['max_radius'])
            except ValueError:
                max_radius = None

        min_area = None
        if 'min_area' in last_row and last_row['min_area'] != '':
            try:
                min_area = float(last_row['min_area'])
            except ValueError:
                min_area = None

        max_area = None
        if 'max_area' in last_row and last_row['max_area'] != '':
            try:
                max_area = float(last_row['max_area'])
            except ValueError:
                max_area = None

        return {
            'noise': float(last_row['noise']),
            'scale': float(last_row['scale']),
            'scale_percent': scale_percent,
            'contrast': float(last_row['contrast']),
            'min_radius': min_radius,
            'max_radius': max_radius,
            'min_area': min_area,
            'max_area': max_area,
        }


def annotate_all_images():
    """Annotate all images in src/data/images using the last parameters from settings.csv"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Get settings file
    settings_file = script_dir / "data" / "settings.csv"
    
    # Get parameters from last row in settings.csv
    try:
        params = get_last_parameters_from_csv(settings_file)
        print(f"Using parameters from settings.csv:")
        print(f"  Noise: {params['noise']:.4f}")
        if params['scale_percent'] is None:
            print(f"  Scale: {params['scale']:.4f}")
        else:
            print(f"  Scale: {params['scale_percent']:.2f}% of image min dimension")
        print(f"  Contrast: {params['contrast']:.4f}")
        min_radius = params['min_radius']
        max_radius = params['max_radius']
        min_area = params['min_area']
        max_area = params['max_area']

        if min_radius is None and min_area is not None:
            min_radius = math.sqrt(min_area / math.pi)
        if max_radius is None and max_area is not None:
            max_radius = math.sqrt(max_area / math.pi)

        if min_radius is None:
            print("  Min radius: auto (from noise)")
        else:
            print(f"  Min radius: {min_radius:.2f}px")
        if max_radius is None:
            print("  Max radius: none")
        else:
            print(f"  Max radius: {max_radius:.2f}px")
    except Exception as e:
        print(f"Error reading parameters: {e}")
        return
    
    # Get input and output directories
    input_dir = script_dir / "data" / "images"
    output_root = script_dir / "data" / "auto_images"
    annotations_dir = output_root / "annotations"
    masks_dir = output_root / "masks"
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # Process each image
    for i, image_file in enumerate(sorted(image_files), 1):
        try:
            # Generate output filename
            output_file = annotations_dir / f"{image_file.stem}_annotated.png"
            mask_file = masks_dir / f"{image_file.stem}_mask.png"

            # Apply annotation
            scale = params['scale']
            if params['scale_percent'] is not None:
                img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    height, width = img.shape[:2]
                    scale = min(width, height) * (params['scale_percent'] / 100.0)

            min_area = params['min_area']
            max_area = params['max_area']
            if params['min_radius'] is not None:
                min_area = math.pi * params['min_radius'] * params['min_radius']
            if params['max_radius'] is not None:
                max_area = math.pi * params['max_radius'] * params['max_radius']

            draw_local_contrast_dark_contours(
                str(image_file),
                noise=params['noise'],
                scale=scale,
                contrast=params['contrast'],
                min_area=min_area,
                max_area=max_area,
                out_path=str(output_file),
                mask_out_path=str(mask_file),
            )
            
            print(
                f"  [{i}/{len(image_files)}] Processed: {image_file.name} -> {output_file.name}, {mask_file.name}"
            )
            
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
    
    print(f"\nFinished! Annotated images saved to {annotations_dir}")
    print(f"Masks saved to {masks_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI for contour annotation"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate all images in src/data/images using last parameters from settings.csv"
    )
    
    args = parser.parse_args()
    
    # Launch GUI if requested
    if args.gui:
        gui_main()
        return
    
    # Annotate all images if requested
    if args.annotate:
        annotate_all_images()
        return
    
    # Extract frames if video is provided
    if args.video:
        # Get the script directory and construct output path
        script_dir = Path(__file__).parent
        output_dir = script_dir / "data" / "images"
        
        # Extract frames
        extract_frames(args.video, str(output_dir), max_frames=MAX_FRAMES)
    else:
        # If no video and no gui flag, show help or launch GUI
        parser.print_help()


if __name__ == "__main__":
    main()
