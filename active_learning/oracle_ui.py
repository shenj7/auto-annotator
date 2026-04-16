import os
import glob
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
import cv2
import numpy as np


class OracleUI:
    """
    Interactive Human Oracle Interface.

    Shows the original TEM image alongside a proposal overlay where each
    detected contour can be individually toggled on (green) or off (red)
    by clicking it.  Accept saves a mask built from only the selected
    contours; Reject discards the image entirely.

    Keyboard shortcuts:  ↑ = Accept   ↓ = Reject
    """

    COLOR_SELECTED   = (0, 220, 0)    # BGR green — contour will be kept
    COLOR_DESELECTED = (0, 0, 220)    # BGR red   — contour will be dropped

    def __init__(self, review_dir: str, verified_dir: str, rejected_dir: str):
        self.review_dir   = Path(review_dir)
        self.verified_dir = Path(verified_dir)
        self.rejected_dir = Path(rejected_dir)

        for d in [self.verified_dir]:
            d.mkdir(parents=True, exist_ok=True)
            (d / "images").mkdir(exist_ok=True)
            (d / "masks").mkdir(exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        (self.rejected_dir / "images").mkdir(exist_ok=True)
        (self.rejected_dir / "masks").mkdir(exist_ok=True)

        # Per-image state
        self._current_img_path  = None   # Path to the original image
        self._img_gray          = None   # (H, W) uint8
        self._contours          = []     # list of np arrays, one per contour
        self._selected          = []     # list of bool, parallel to _contours
        self._display_scale     = 1.0    # canvas-px / image-px ratio
        self._display_offset    = (0, 0) # (x, y) top-left of image on canvas

        self._fill_screen   = False  # False = cap at 2×, True = fill canvas

        self.root           = None
        self.orig_photo     = None
        self.overlay_photo  = None
        self._resize_timer  = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_next_item(self) -> bool:
        images_dir  = self.review_dir / "images"
        image_paths = sorted(p for p in images_dir.glob("*.*")
                             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        if not image_paths:
            self._current_img_path = None
            self._img_gray = None
            self._contours = []
            self._selected = []
            return False

        self._current_img_path = image_paths[0]
        self._img_gray = cv2.imread(str(self._current_img_path), cv2.IMREAD_GRAYSCALE)

        # Load the best available mask and extract contours from it.
        # Prefer the middle threshold (index 1); fall back to whatever exists.
        stem       = self._current_img_path.stem
        mask_paths = sorted((self.review_dir / "masks").glob(f"{stem}*.*"))
        mask       = None
        if mask_paths:
            mid = len(mask_paths) // 2
            mask = cv2.imread(str(mask_paths[mid]), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self._contours = list(contours)
        else:
            self._contours = []

        # All contours start selected
        self._selected = [True] * len(self._contours)
        return True

    def _cleanup_current(self):
        """Remove all pending_review files for the current image."""
        if self._current_img_path and self._current_img_path.exists():
            os.remove(self._current_img_path)
        stem = self._current_img_path.stem if self._current_img_path else ""
        for subdir in ("masks", "overlays"):
            for p in (self.review_dir / subdir).glob(f"{stem}*.*"):
                p.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_overlay(self) -> np.ndarray:
        """Draw contours onto the image — green if selected, red if not."""
        img_bgr = cv2.cvtColor(self._img_gray, cv2.COLOR_GRAY2BGR)
        for contour, selected in zip(self._contours, self._selected):
            color = self.COLOR_SELECTED if selected else self.COLOR_DESELECTED
            cv2.drawContours(img_bgr, [contour], -1, color, 2)
        return img_bgr

    def _render_overlay(self):
        """Redraw the overlay canvas with current selection state."""
        if self._img_gray is None:
            return
        overlay_bgr = self._build_overlay()
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        pil_img     = Image.fromarray(overlay_rgb)

        self.root.update_idletasks()
        cw = self.overlay_canvas.winfo_width()  or 500
        ch = self.overlay_canvas.winfo_height() or 500
        ih, iw = self._img_gray.shape
        scale = min(cw / iw, ch / ih) if self._fill_screen else min(cw / iw, ch / ih, 2.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        self._display_scale  = scale
        self._display_offset = ((cw - nw) // 2, (ch - nh) // 2)

        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.overlay_photo = ImageTk.PhotoImage(pil_img)
        self.overlay_canvas.delete("all")
        ox, oy = self._display_offset
        self.overlay_canvas.create_image(ox, oy, anchor=tk.NW, image=self.overlay_photo)

        n_sel = sum(self._selected)
        self.selection_label.config(
            text=f"{n_sel} / {len(self._contours)} contours selected"
        )

    def _render_original(self):
        if self._img_gray is None:
            return
        pil_img = Image.fromarray(self._img_gray)
        self.root.update_idletasks()
        cw = self.orig_canvas.winfo_width()  or 500
        ch = self.orig_canvas.winfo_height() or 500
        ih, iw = self._img_gray.shape
        scale = min(cw / iw, ch / ih) if self._fill_screen else min(cw / iw, ch / ih, 2.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.orig_photo = ImageTk.PhotoImage(pil_img)
        self.orig_canvas.delete("all")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.orig_canvas.create_image(ox, oy, anchor=tk.NW, image=self.orig_photo)

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        """Toggle the contour under the click, if any."""
        if not self._contours:
            return

        # Convert canvas coords → image coords
        ox, oy = self._display_offset
        scale  = self._display_scale
        img_x  = (event.x - ox) / scale
        img_y  = (event.y - oy) / scale

        # Tolerance in image-space pixels: 10 canvas px converted back to image px.
        # This makes small contours clickable even if the click misses the interior.
        tolerance = 10.0 / scale

        # Find the contour closest to the click (by distance to boundary).
        # pointPolygonTest with measureDist=True returns:
        #   positive = inside (value is distance to nearest edge)
        #   negative = outside (abs value is distance to nearest edge)
        # We pick the contour whose boundary is nearest to the click,
        # as long as it's within tolerance. For ties we prefer the smaller contour.
        best_idx  = None
        best_dist = float('inf')
        best_area = float('inf')
        for i, contour in enumerate(self._contours):
            dist = cv2.pointPolygonTest(contour, (float(img_x), float(img_y)), measureDist=True)
            boundary_dist = abs(dist)  # distance to nearest edge regardless of inside/outside
            area = cv2.contourArea(contour)
            if boundary_dist <= tolerance:
                if boundary_dist < best_dist or (boundary_dist == best_dist and area < best_area):
                    best_dist = boundary_dist
                    best_area = area
                    best_idx  = i

        if best_idx is not None:
            self._selected[best_idx] = not self._selected[best_idx]
            self._render_overlay()

    # ------------------------------------------------------------------
    # Accept / Reject
    # ------------------------------------------------------------------

    def on_accept(self):
        if self._current_img_path is None or self._img_gray is None:
            return
        try:
            stem = self._current_img_path.stem
            h, w = self._img_gray.shape

            # Build mask from selected contours only
            mask = np.zeros((h, w), dtype=np.uint8)
            selected_contours = [c for c, s in zip(self._contours, self._selected) if s]
            if selected_contours:
                cv2.drawContours(mask, selected_contours, -1, 255, thickness=cv2.FILLED)

            cv2.imwrite(str(self.verified_dir / "images" / f"{stem}.png"), self._img_gray)
            cv2.imwrite(str(self.verified_dir / "masks"  / f"{stem}.png"), mask)

            self._cleanup_current()
            self.load_interface_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error accepting image: {e}")

    def on_reject(self):
        if self._current_img_path is None:
            return
        try:
            stem = self._current_img_path.stem

            # Save image + current (possibly partial) mask for contrastive training
            (self.rejected_dir / "images").mkdir(exist_ok=True)
            (self.rejected_dir / "masks").mkdir(exist_ok=True)
            cv2.imwrite(str(self.rejected_dir / "images" / f"{stem}.png"), self._img_gray)

            if self._contours:
                h, w = self._img_gray.shape
                rej_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(rej_mask, self._contours, -1, 255, thickness=cv2.FILLED)
                cv2.imwrite(str(self.rejected_dir / "masks" / f"{stem}.png"), rej_mask)

            self._cleanup_current()
            self.load_interface_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error rejecting image: {e}")

    # ------------------------------------------------------------------
    # Interface loading
    # ------------------------------------------------------------------

    def load_interface_data(self):
        has_items = self._get_next_item()

        if not has_items:
            self.status_label.config(text="Status: No more images in the queue! Great job.")
            self.orig_canvas.delete("all")
            self.overlay_canvas.delete("all")
            self.accept_btn.config(state=tk.DISABLED)
            self.reject_btn.config(state=tk.DISABLED)
            self.selection_label.config(text="")
            return

        self.status_label.config(
            text=f"Reviewing: {self._current_img_path.name}  |  "
                 f"{len(self._contours)} contours proposed"
        )
        self.accept_btn.config(state=tk.NORMAL)
        self.reject_btn.config(state=tk.NORMAL)
        self._render_original()
        self._render_overlay()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _toggle_fill_screen(self):
        self._fill_screen = not self._fill_screen
        self.zoom_btn.config(text="Zoom: fill screen" if self._fill_screen else "Zoom: 2× cap")
        self._redraw_images()

    def launch(self, **kwargs):
        self.root = tk.Tk()
        self.root.title("Dislocation Boundary Oracle")
        self.root.geometry("1200x600")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Header
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Label(top_frame, text="Dislocation Boundaries — Human Oracle Interface",
                  font=("Helvetica", 16, "bold")).pack(anchor=tk.W)
        self.status_label = ttk.Label(top_frame, text="Status: Initializing...",
                                      font=("Helvetica", 12))
        self.status_label.pack(anchor=tk.W, pady=(5, 0))

        # Content
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.columnconfigure(2, weight=0)
        content_frame.rowconfigure(0, weight=1)

        # Original image
        orig_frame = ttk.LabelFrame(content_frame, text="Original TEM Image")
        orig_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self.orig_canvas = tk.Canvas(orig_frame, bg="gray")
        self.orig_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Interactive overlay canvas
        overlay_frame = ttk.LabelFrame(content_frame,
                                       text="Proposed Contours  —  click to toggle on/off")
        overlay_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self.overlay_canvas = tk.Canvas(overlay_frame, bg="gray", cursor="crosshair")
        self.overlay_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.overlay_canvas.bind("<Button-1>", self._on_canvas_click)

        # Controls
        controls_frame = ttk.Frame(content_frame, width=220)
        controls_frame.grid(row=0, column=2, sticky=(tk.N, tk.E, tk.S), padx=5)
        controls_frame.pack_propagate(False)

        legend_frame = ttk.LabelFrame(controls_frame, text="Legend")
        legend_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(legend_frame, text="  Selected (keep)",
                 bg="#00DC00", fg="black", anchor=tk.W).pack(fill=tk.X, padx=5, pady=2)
        tk.Label(legend_frame, text="  Deselected (drop)",
                 bg="#DC0000", fg="white", anchor=tk.W).pack(fill=tk.X, padx=5, pady=2)

        self.selection_label = ttk.Label(controls_frame, text="", font=("Helvetica", 11))
        self.selection_label.pack(pady=(0, 10))

        # Fill-screen toggle
        self.zoom_btn = tk.Button(
            controls_frame, text="Zoom: 2× cap",
            font=("Helvetica", 10), command=self._toggle_fill_screen)
        self.zoom_btn.pack(fill=tk.X, padx=5, pady=(0, 15))

        actions_frame = ttk.LabelFrame(controls_frame, text="Actions")
        actions_frame.pack(fill=tk.X)
        self.accept_btn = tk.Button(
            actions_frame, text="✅ Accept Selected [↑]",
            bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
            command=self.on_accept)
        self.accept_btn.pack(fill=tk.X, padx=5, pady=10, ipady=5)
        self.reject_btn = tk.Button(
            actions_frame, text="❌ Reject All [↓]",
            bg="#F44336", fg="white", font=("Helvetica", 12, "bold"),
            command=self.on_reject)
        self.reject_btn.pack(fill=tk.X, padx=5, pady=5, ipady=5)

        # Keyboard shortcuts
        self.root.bind('<Up>',   lambda e: self.on_accept())
        self.root.bind('<Down>', lambda e: self.on_reject())

        # Resize debounce
        self.root.bind('<Configure>', self._on_resize)
        self._resize_timer = None

        self.root.after(100, self.load_interface_data)
        self.root.mainloop()

    def _on_resize(self, event):
        if event.widget == self.root:
            if self._resize_timer:
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(200, self._redraw_images)

    def _redraw_images(self):
        if self._img_gray is not None:
            self._render_original()
            self._render_overlay()


if __name__ == "__main__":
    ui = OracleUI(
        review_dir="active_learning_data/pending_review",
        verified_dir="active_learning_data/verified_dataset",
        rejected_dir="active_learning_data/rejected"
    )
    ui.launch()
