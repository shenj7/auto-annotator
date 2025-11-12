import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from annotation_utils import draw_local_contrast_dark_contours
import tempfile
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
        
        # Scale slider (0.0 to 20.0)
        ttk.Label(params_frame, text="Scale (0.0 - 20.0):").pack(anchor=tk.W, pady=(10, 5))
        self.scale_var = tk.DoubleVar(value=5.0)
        scale_scale = ttk.Scale(params_frame, from_=0.0, to=20.0, variable=self.scale_var,
                               orient=tk.HORIZONTAL, command=self.on_slider_change)
        scale_scale.pack(fill=tk.X, pady=(0, 5))
        self.scale_label = ttk.Label(params_frame, text="5.0")
        self.scale_label.pack(anchor=tk.W)
        self.scale_var.trace('w', lambda *args: self.scale_label.config(text=f"{self.scale_var.get():.2f}"))
        
        # Contrast slider (0 to 255)
        ttk.Label(params_frame, text="Contrast (0 - 255):").pack(anchor=tk.W, pady=(10, 5))
        self.contrast_var = tk.DoubleVar(value=0.0)
        contrast_scale = ttk.Scale(params_frame, from_=0.0, to=255.0, variable=self.contrast_var,
                                  orient=tk.HORIZONTAL, command=self.on_slider_change)
        contrast_scale.pack(fill=tk.X, pady=(0, 5))
        self.contrast_label = ttk.Label(params_frame, text="0")
        self.contrast_label.pack(anchor=tk.W)
        self.contrast_var.trace('w', lambda *args: self.contrast_label.config(text=f"{int(self.contrast_var.get())}"))
        
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
            self.update_image()
    
    def on_slider_change(self, *args):
        """Called when any slider changes"""
        if self.current_image_path:
            self.update_image()
    
    def update_image(self):
        """Update the displayed image with current parameters"""
        if not self.current_image_path:
            return
        
        try:
            # Get current parameter values
            noise = self.noise_var.get()
            scale = self.scale_var.get()
            contrast = self.contrast_var.get()
            
            # Call the contour function
            draw_local_contrast_dark_contours(
                self.current_image_path,
                noise=noise,
                scale=scale,
                contrast=contrast,
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
        scale = self.scale_var.get()
        contrast = self.contrast_var.get()
        
        # Check if file exists and has headers
        file_exists = settings_file.exists()
        has_headers = False
        
        if file_exists:
            try:
                with open(settings_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    first_line = next(reader, None)
                    has_headers = first_line is not None and first_line[0].lower() == 'image_path'
            except:
                pass
        
        # Write headers if file doesn't exist or doesn't have headers
        write_headers = not file_exists or not has_headers
        
        try:
            with open(settings_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write headers if needed
                if write_headers:
                    writer.writerow(['image_path', 'noise', 'scale', 'contrast', 'timestamp'])
                
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
                
                writer.writerow([image_path, f"{noise:.4f}", f"{scale:.4f}", f"{contrast:.4f}", timestamp])
                
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

