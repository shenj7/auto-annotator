import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from annotation_utils import draw_local_contrast_dark_contours, min_area_from_noise
import tempfile
import math
import os
import csv
import shutil
from datetime import datetime


class ContourGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Contour Annotation Tool")
        self.root.geometry("1200x800")
        
        self.current_image_path = None
        self.temp_output_path = None
        self.temp_mask_path = None
        self._image_min_dim = None
        self._radius_min_limit = 1
        self._radius_max_limit = 5000
        self._suppress_slider_callback = False
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="Image Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(file_frame, text="Select Image", command=self.select_image).pack(fill=tk.X)
        self.image_label = ttk.Label(file_frame, text="No image selected", foreground="gray")
        self.image_label.pack(pady=(5, 0))
        
        # Parameters frame
        params_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="10")
        params_frame.pack(fill=tk.X)
        
        # Noise/Sensitivity slider (0.0 to 1.0)
        ttk.Label(params_frame, text="Noise/Sensitivity (0.0 - 1.0):").pack(anchor=tk.W, pady=(0, 5))
        self.noise_var = tk.DoubleVar(value=0.45)
        noise_scale = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.noise_var, 
                               orient=tk.HORIZONTAL, command=self.on_slider_change)
        noise_scale.pack(fill=tk.X, pady=(0, 5))
        self.noise_label = ttk.Label(params_frame, text="0.45")
        self.noise_label.pack(anchor=tk.W)
        self.noise_var.trace('w', lambda *args: self.noise_label.config(text=f"{self.noise_var.get():.2f}"))
        
        # Scale radius percent (0.1% to 2% of image)
        ttk.Label(params_frame, text="Scale Radius (% of image, 0.1 - 2):").pack(anchor=tk.W, pady=(10, 5))
        self.scale_percent_var = tk.DoubleVar(value=2.0)
        scale_scale = ttk.Scale(params_frame, from_=0.1, to=2.0, variable=self.scale_percent_var,
                               orient=tk.HORIZONTAL, command=self.on_slider_change)
        scale_scale.pack(fill=tk.X, pady=(0, 5))
        self.scale_label = ttk.Label(params_frame, text="2.00%")
        self.scale_label.pack(anchor=tk.W)
        self.scale_percent_var.trace(
            'w',
            lambda *args: self.scale_label.config(text=f"{self.scale_percent_var.get():.2f}%"),
        )
        
        # Contrast slider (0 to 255)
        ttk.Label(params_frame, text="Contrast (0 - 255):").pack(anchor=tk.W, pady=(10, 5))
        self.contrast_var = tk.DoubleVar(value=0.0)
        contrast_scale = ttk.Scale(params_frame, from_=0.0, to=255.0, variable=self.contrast_var,
                                  orient=tk.HORIZONTAL, command=self.on_slider_change)
        contrast_scale.pack(fill=tk.X, pady=(0, 5))
        self.contrast_label = ttk.Label(params_frame, text="0")
        self.contrast_label.pack(anchor=tk.W)
        self.contrast_var.trace('w', lambda *args: self.contrast_label.config(text=f"{int(self.contrast_var.get())}"))

        # Min radius slider (0.01% to 1% of image size)
        ttk.Label(params_frame, text="Min Radius (pixels, 0.01% - 1% of image):").pack(anchor=tk.W, pady=(10, 5))
        default_min_area = min_area_from_noise(self.noise_var.get())
        default_min_radius = math.sqrt(default_min_area / math.pi)
        self.min_radius_var = tk.DoubleVar(value=float(default_min_radius))
        self.min_radius_pos_var = tk.DoubleVar(
            value=self._pos_from_radius(default_min_radius)
        )
        self.min_radius_scale = ttk.Scale(
            params_frame,
            from_=0.0,
            to=1.0,
            variable=self.min_radius_pos_var,
            orient=tk.HORIZONTAL,
            command=self._on_min_radius_slider,
        )
        self.min_radius_scale.pack(fill=tk.X, pady=(0, 5))
        self.min_radius_label = ttk.Label(params_frame, text=str(int(default_min_radius)))
        self.min_radius_label.pack(anchor=tk.W)
        self.min_radius_var.trace('w', lambda *args: self.min_radius_label.config(text=f"{int(self.min_radius_var.get())}"))

        # Max radius slider (0.01% to 1% of image size)
        ttk.Label(params_frame, text="Max Radius (pixels, 0.01% - 1% of image):").pack(anchor=tk.W, pady=(10, 5))
        self.max_radius_var = tk.DoubleVar(value=float(self._radius_max_limit))
        self.max_radius_pos_var = tk.DoubleVar(value=1.0)
        self.max_radius_scale = ttk.Scale(
            params_frame,
            from_=0.0,
            to=1.0,
            variable=self.max_radius_pos_var,
            orient=tk.HORIZONTAL,
            command=self._on_max_radius_slider,
        )
        self.max_radius_scale.pack(fill=tk.X, pady=(0, 5))
        self.max_radius_label = ttk.Label(params_frame, text=str(int(self._radius_max_limit)))
        self.max_radius_label.pack(anchor=tk.W)
        self.max_radius_var.trace('w', lambda *args: self.max_radius_label.config(text=f"{int(self.max_radius_var.get())}"))

        # Save button
        save_frame = ttk.Frame(control_frame)
        save_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(save_frame, text="Save Result", command=self.save_result).pack(fill=tk.X)
        
        # Right panel for image display
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Canvas with scrollbars for image
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Create temp file for output
        self.temp_output_path = os.path.join(tempfile.gettempdir(), "contour_temp_output.png")
        self.temp_mask_path = os.path.join(tempfile.gettempdir(), "contour_temp_mask.png")
        
        # Try to load default image if available
        self.load_default_image()
    
    def load_default_image(self):
        """Try to load the first frame from data/images if available"""
        script_dir = Path(__file__).parent
        default_image = script_dir / "data" / "images" / "frame_000000.jpg"
        if default_image.exists():
            self.current_image_path = str(default_image)
            self.image_label.config(text=os.path.basename(self.current_image_path), foreground="black")
            self._update_radius_limits()
            self.update_image()
    
    def select_image(self):
        """Open file dialog to select an image"""
        initial_dir = Path(__file__).parent / "data" / "images"
        if not initial_dir.exists():
            initial_dir = Path.home()
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=str(initial_dir),
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=os.path.basename(file_path), foreground="black")
            self._update_radius_limits()
            self.update_image()
    
    def on_slider_change(self, *args):
        """Called when any slider changes"""
        if self._suppress_slider_callback:
            return
        self._enforce_radius_bounds()
        if self.current_image_path:
            self.update_image()

    def _radius_limits_from_image(self):
        if not self.current_image_path:
            return None
        try:
            with Image.open(self.current_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image size: {e}")
            return None
        base = min(width, height)
        self._image_min_dim = base
        min_limit = max(1, int(round(base * 0.0001)))
        max_limit = max(min_limit, int(round(base * 0.01)))
        return min_limit, max_limit

    def _pos_from_radius(self, radius):
        min_val = float(self._radius_min_limit)
        max_val = float(self._radius_max_limit)
        if max_val <= min_val:
            return 0.0
        if radius <= 0:
            return 0.0
        radius = max(min_val, min(radius, max_val))
        return math.log(radius / min_val) / math.log(max_val / min_val)

    def _radius_from_pos(self, pos):
        min_val = float(self._radius_min_limit)
        max_val = float(self._radius_max_limit)
        if max_val <= min_val:
            return min_val
        pos = max(0.0, min(1.0, pos))
        return min_val * ((max_val / min_val) ** pos)

    def _update_radius_limits(self):
        limits = self._radius_limits_from_image()
        if limits is None:
            return
        previous_min = self._radius_min_limit
        previous_max = self._radius_max_limit
        self._radius_min_limit, self._radius_max_limit = limits
        self.min_radius_scale.configure(from_=0.0, to=1.0)
        self.max_radius_scale.configure(from_=0.0, to=1.0)
        if int(round(self.min_radius_var.get())) == previous_min:
            self.min_radius_var.set(float(self._radius_min_limit))
        if int(round(self.max_radius_var.get())) == previous_max:
            self.max_radius_var.set(float(self._radius_max_limit))
        self._enforce_radius_bounds()

    def _enforce_radius_bounds(self):
        min_val = self.min_radius_var.get()
        max_val = self.max_radius_var.get()
        min_val = max(float(self._radius_min_limit), min(min_val, float(self._radius_max_limit)))
        max_val = max(float(self._radius_min_limit), min(max_val, float(self._radius_max_limit)))
        if max_val < min_val:
            max_val = min_val
        self._suppress_slider_callback = True
        self.min_radius_var.set(min_val)
        self.max_radius_var.set(max_val)
        self.min_radius_pos_var.set(self._pos_from_radius(min_val))
        self.max_radius_pos_var.set(self._pos_from_radius(max_val))
        self._suppress_slider_callback = False

    def _on_min_radius_slider(self, *args):
        if self._suppress_slider_callback:
            return
        self.min_radius_var.set(self._radius_from_pos(self.min_radius_pos_var.get()))
        self._enforce_radius_bounds()
        if self.current_image_path:
            self.update_image()

    def _on_max_radius_slider(self, *args):
        if self._suppress_slider_callback:
            return
        self.max_radius_var.set(self._radius_from_pos(self.max_radius_pos_var.get()))
        self._enforce_radius_bounds()
        if self.current_image_path:
            self.update_image()

    def _scale_pixels_from_percent(self, percent):
        min_dim = self._image_min_dim
        if min_dim is None:
            self._radius_limits_from_image()
            min_dim = self._image_min_dim
        if min_dim is None:
            return None
        return float(min_dim) * (percent / 100.0)
    
    def update_image(self):
        """Update the displayed image with current parameters"""
        if not self.current_image_path:
            return
        
        try:
            # Get current parameter values
            noise = self.noise_var.get()
            scale_percent = self.scale_percent_var.get()
            scale = self._scale_pixels_from_percent(scale_percent)
            if scale is None:
                scale = 1.0
            contrast = self.contrast_var.get()
            min_radius = self.min_radius_var.get()
            max_radius = self.max_radius_var.get()
            min_area = math.pi * min_radius * min_radius
            max_area = math.pi * max_radius * max_radius
            
            # Call the contour function
            draw_local_contrast_dark_contours(
                self.current_image_path,
                noise=noise,
                scale=scale,
                contrast=contrast,
                min_area=min_area,
                max_area=max_area,
                out_path=self.temp_output_path,
                mask_out_path=self.temp_mask_path
            )
            
            # Load and display the result
            img = Image.open(self.temp_output_path)
            self.display_image(img)
            
        except Exception as e:
            print(f"Error updating image: {e}")
    
    def display_image(self, img):
        """Display image in the canvas"""
        # Calculate scaling to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet sized, schedule update
            self.root.after(100, lambda: self.display_image(img))
            return
        
        img_width, img_height = img.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_resized)
        
        # Clear canvas and add image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def save_settings_to_csv(self):
        """Append current parameter settings to settings.csv"""
        script_dir = Path(__file__).parent
        project_root = script_dir.parent  # Parent of src directory
        settings_file = script_dir / "data" / "settings.csv"
        
        # Ensure data directory exists
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current parameter values
        noise = self.noise_var.get()
        scale_percent = self.scale_percent_var.get()
        scale = self._scale_pixels_from_percent(scale_percent)
        if scale is None:
            scale = 1.0
        contrast = self.contrast_var.get()
        min_radius = float(self.min_radius_var.get())
        max_radius = float(self.max_radius_var.get())

        fieldnames = [
            'image_path',
            'noise',
            'scale',
            'scale_percent',
            'contrast',
            'min_radius',
            'max_radius',
            'timestamp',
        ]
        existing_fieldnames = None

        if settings_file.exists():
            try:
                with open(settings_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    existing_fieldnames = next(reader, None)
            except Exception:
                existing_fieldnames = None

        if existing_fieldnames and existing_fieldnames != fieldnames:
            try:
                with open(settings_file, 'r', newline='') as f:
                    dict_reader = csv.DictReader(f)
                    existing_rows = list(dict_reader)
                for row in existing_rows:
                    if not row.get('min_radius'):
                        min_area_value = row.get('min_area', '')
                        if min_area_value:
                            try:
                                area = float(min_area_value)
                                row['min_radius'] = f"{math.sqrt(area / math.pi):.2f}"
                            except ValueError:
                                row['min_radius'] = ''
                        else:
                            noise_value = row.get('noise', '')
                            if noise_value:
                                try:
                                    area = min_area_from_noise(float(noise_value))
                                    row['min_radius'] = f"{math.sqrt(area / math.pi):.2f}"
                                except ValueError:
                                    row['min_radius'] = ''
                            else:
                                row['min_radius'] = ''
                    if not row.get('max_radius'):
                        max_area_value = row.get('max_area', '')
                        if max_area_value:
                            try:
                                area = float(max_area_value)
                                row['max_radius'] = f"{math.sqrt(area / math.pi):.2f}"
                            except ValueError:
                                row['max_radius'] = ''
                        else:
                            row['max_radius'] = ''
                    if not row.get('scale_percent'):
                        row['scale_percent'] = ''
                with open(settings_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                existing_fieldnames = fieldnames
            except Exception as e:
                print(f"Error migrating settings.csv: {e}")
                existing_fieldnames = None

        write_headers = not settings_file.exists() or not existing_fieldnames

        try:
            with open(settings_file, 'a', newline='') as f:
                writer = csv.writer(f)

                if write_headers:
                    writer.writerow(fieldnames)

                # Write current settings
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Convert absolute path to relative path from project root
                if self.current_image_path:
                    try:
                        image_path_abs = Path(self.current_image_path).resolve()
                        image_path_rel = image_path_abs.relative_to(project_root.resolve())
                        image_path = str(image_path_rel).replace('\\', '/')  # Use forward slashes
                    except ValueError:
                        # If path is not under project root, use absolute path as fallback
                        image_path = self.current_image_path
                else:
                    image_path = ''

                writer.writerow([
                    image_path,
                    f"{noise:.4f}",
                    f"{scale:.4f}",
                    f"{scale_percent:.2f}",
                    f"{contrast:.4f}",
                    f"{min_radius:.2f}",
                    f"{max_radius:.2f}",
                    timestamp,
                ])

        except Exception as e:
            print(f"Error saving settings to CSV: {e}")
    
    def save_result(self):
        """Save the current result automatically to src/data/annotated_images"""
        if not self.current_image_path:
            tk.messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        try:
            # Get the script directory and construct output path
            script_dir = Path(__file__).parent
            output_dir = script_dir / "data" / "annotated_images"
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename based on input image name
            input_path = Path(self.current_image_path)
            base_name = input_path.stem
            output_file = output_dir / f"{base_name}_annotated.png"
            mask_output_file = output_dir / f"{base_name}_mask.png"
            
            # Copy the annotated image to the output directory
            shutil.copy(self.temp_output_path, str(output_file))

            # Copy the mask image if available
            mask_saved = False
            if self.temp_mask_path and os.path.exists(self.temp_mask_path):
                shutil.copy(self.temp_mask_path, str(mask_output_file))
                mask_saved = True
            
            # Append settings to CSV
            self.save_settings_to_csv()
            
            message_lines = [f"Result saved to {output_file}"]
            if mask_saved:
                message_lines.append(f"Mask saved to {mask_output_file}")
            message_lines.append("Settings saved to settings.csv")

            tk.messagebox.showinfo("Success", "\n".join(message_lines))
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save: {e}")


def main():
    root = tk.Tk()
    app = ContourGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
