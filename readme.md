This tool allows you to adjust automatic annotation parameters to improve automatic annotations moving forward

Structure: data/auto_images has automatically annotated images
data/annotated_images has manually adjusted annotations
data/masks/tracking has loop stats


How to use:
1. Add your video to src/data
2. Extract your video using src/app.py --video path/to/your/video
3. Run the GUI with python src/app.py --gui
4. When you're happy with your parameters, run python src/app.py --annotate to annotate the rest of the images with your last used parameters. Annotated images and companion masks (white background, black foreground) are written to `src/data/auto_images/annotations` and `src/data/auto_images/masks` respectively.
5. Get colorized images and tracks with python3 src/mask_tracker.py --mask-dir src/data/auto_images/masks --out-dir src/data/auto_images/masks/tracking
