import argparse
from pathlib import Path
import csv
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
        
        return {
            'noise': float(last_row['noise']),
            'scale': float(last_row['scale']),
            'contrast': float(last_row['contrast'])
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
        print(f"  Scale: {params['scale']:.4f}")
        print(f"  Contrast: {params['contrast']:.4f}")
    except Exception as e:
        print(f"Error reading parameters: {e}")
        return
    
    # Get input and output directories
    input_dir = script_dir / "data" / "images"
    output_dir = script_dir / "data" / "auto_images"
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            output_file = output_dir / f"{image_file.stem}_annotated.png"
            
            # Apply annotation
            draw_local_contrast_dark_contours(
                str(image_file),
                noise=params['noise'],
                scale=params['scale'],
                contrast=params['contrast'],
                out_path=str(output_file)
            )
            
            print(f"  [{i}/{len(image_files)}] Processed: {image_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
    
    print(f"\nFinished! Annotated images saved to {output_dir}")


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

