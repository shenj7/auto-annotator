"""
track.py — Run the trained model over a sequence of TEM images, produce masks,
track dislocations across frames, and output statistics.

Usage:
    python active_learning/track.py --image-dir src/data/images/

    # With optional grain boundaries and custom output dir:
    python active_learning/track.py \\
        --image-dir src/data/images/ \\
        --out-dir   results/tracking/ \\
        --gb-path   grain_boundaries.json \\
        --threshold 0.5

Outputs (written to --out-dir):
    masks/          — binary mask PNG for each frame
    overlays/       — original image + green contours overlay
    tracks/         — coloured blobs with track IDs and trajectory lines
    colorized/      — coloured blobs only (no labels)
    tracks.csv      — per-detection table (frame, track_id, area, centroid, ...)
    loop_stats.csv  — per-frame aggregate (loop count, mean size, mean diameter)
"""

import argparse
import csv
import json
import sys
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from model import ResAttentionUNet
from patch_utils import extract_patches, merge_patches
from geometry_utils import min_dist_to_grain_boundaries

# ---------------------------------------------------------------------------
# Constants (match src/mask_tracker.py defaults)
# ---------------------------------------------------------------------------

PATCH_SIZE      = 128
CHECKPOINT_PATH = Path("active_learning_data/checkpoints/model.pt")

MIN_AREA       = 5
MAX_DISTANCE   = 8.0    # per-frame matching radius (px) — dislocations barely move
MIN_ROUNDNESS  = 0.6
MAX_AGE        = 15     # frames before an active track is moved to graveyard
GRAVEYARD_AGE  = 40     # frames a dead track stays revivable in the graveyard
BG_DILATE_ITER = 3
FG_THRESH_REL  = 0.35   # fraction of max distance-transform value

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

def load_model(device: str) -> ResAttentionUNet:
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}.")
        print("Train first:  python active_learning/main.py --train")
        sys.exit(1)
    model = ResAttentionUNet(n_channels=1, n_classes=1, dropout_rate=0.3)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded checkpoint — round {ckpt.get('round','?')}, "
          f"{ckpt.get('num_verified','?')} verified samples.")
    return model


def predict_mask(
    model: ResAttentionUNet,
    img_gray: np.ndarray,
    device: str,
    threshold: float = 0.5,
    n_passes: int = 1,
) -> np.ndarray:
    """
    Run the model on a full image (patch → stitch) and return a uint8 binary
    mask (255 = dislocation, 0 = background).

    n_passes=1 is fastest; increase for smoother probability maps at the cost
    of inference time.
    """
    h, w = img_gray.shape
    patches = extract_patches(
        img_gray,
        patch_size=(PATCH_SIZE, PATCH_SIZE),
        stride=(PATCH_SIZE, PATCH_SIZE),
    )

    model.eval()
    if n_passes > 1:
        model.enable_dropout()

    prob_patches = []
    with torch.no_grad():
        for patch_img, x, y in patches:
            tensor = (torch.from_numpy(patch_img).float().unsqueeze(0).unsqueeze(0) / 255.0
                      ).to(device)
            preds = []
            for _ in range(n_passes):
                preds.append(torch.sigmoid(model(tensor)))
            mean_p = torch.stack(preds).mean(dim=0).squeeze().cpu().numpy()
            prob_patches.append((mean_p, x, y))

    prob_map = merge_patches(prob_patches, (h, w)).astype(np.float32)
    binary   = (prob_map >= threshold).astype(np.uint8) * 255
    return binary


# ---------------------------------------------------------------------------
# Component detection (watershed — copied from src/mask_tracker.py)
# ---------------------------------------------------------------------------

def _color_from_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(track_id * 9973 + 17)
    return tuple(int(x) for x in rng.randint(60, 255, size=3))


def _extract_frame_index(path: Path, default_idx: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else default_idx


def _find_components(
    mask: np.ndarray,
    fg_thresh_rel: float = FG_THRESH_REL,
) -> Tuple[List[Dict], np.ndarray]:
    """Watershed-based blob detection on a binary mask."""
    fg = (mask > 128).astype(np.uint8)   # model output: bright = dislocation
    if fg.max() == 0:
        return [], np.zeros_like(fg, dtype=np.int32)

    kernel  = np.ones((3, 3), np.uint8)
    opened  = fg * 255
    sure_bg = cv2.dilate(opened, kernel, iterations=BG_DILATE_ITER)

    dist    = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    max_d   = dist.max()
    if max_d <= 0:
        return [], np.zeros_like(fg, dtype=np.int32)

    _, sure_fg = cv2.threshold(dist, fg_thresh_rel * max_d, 255, cv2.THRESH_BINARY)
    sure_fg    = sure_fg.astype(np.uint8)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    labels = markers
    labels[labels == -1] = 0

    comps: List[Dict] = []
    for label_id in np.unique(labels):
        if label_id <= 1:
            continue
        mask_label = (labels == label_id).astype(np.uint8)
        area = int(mask_label.sum())
        if area < MIN_AREA:
            continue
        cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        roundness = float(4.0 * pi * area / (perimeter * perimeter))
        if roundness < MIN_ROUNDNESS:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        roi = mask_label[y:y+bh, x:x+bw]
        m   = cv2.moments(roi)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"]) + x
        cy = float(m["m01"] / m["m00"]) + y
        comps.append({
            "label":   int(label_id),
            "centroid": (cx, cy),
            "bbox":    (int(x), int(y), int(bw), int(bh)),
            "area":    area,
            "contour": cnt,
        })

    return comps, labels


# ---------------------------------------------------------------------------
# Tracker (Hungarian algorithm — copied from src/mask_tracker.py)
# ---------------------------------------------------------------------------

class SimpleTracker:
    """
    Hungarian-algorithm tracker with two-stage matching and a graveyard.

    Stage 1 — match detections to *active* tracks (last seen within max_age frames).
    Stage 2 — match remaining unmatched detections to *dead* tracks in the graveyard
              (last seen within graveyard_age frames).  If a match is found the track
              is revived with its original ID, preserving the full history.
    Stage 3 — any detection still unmatched becomes a brand-new track.

    Distance thresholds scale with the actual frame gap so skipped frames don't
    cause spurious ID switches.
    """

    def __init__(
        self,
        max_distance:   float = MAX_DISTANCE,
        max_age:        int   = MAX_AGE,
        graveyard_age:  int   = GRAVEYARD_AGE,
    ):
        self.max_distance  = float(max_distance)
        self.max_age       = int(max_age)
        self.graveyard_age = int(graveyard_age)

        # Active tracks
        self.active:     Dict[int, Tuple[float, float]] = {}
        self.last_seen:  Dict[int, int]                 = {}
        self.history:    Dict[int, List]                = {}

        # Dead tracks kept for possible revival
        self.graveyard:          Dict[int, Tuple[float, float]] = {}
        self.graveyard_last_seen: Dict[int, int]                = {}

        self.next_id = 1

    def _register(self, det: Dict, frame_idx: int) -> int:
        tid = self.next_id
        self.next_id += 1
        self.active[tid]    = det["centroid"]
        self.last_seen[tid] = frame_idx
        self.history[tid]   = [(frame_idx, *det["centroid"])]
        return tid

    def _revive(self, tid: int, det: Dict, frame_idx: int) -> None:
        """Move a graveyard track back to active."""
        self.active[tid]    = det["centroid"]
        self.last_seen[tid] = frame_idx
        self.history[tid].append((frame_idx, *det["centroid"]))
        self.graveyard.pop(tid, None)
        self.graveyard_last_seen.pop(tid, None)

    @staticmethod
    def _match(
        detections: List[Dict],
        pool: Dict[int, Tuple[float, float]],
        last_seen: Dict[int, int],
        frame_idx: int,
        max_distance: float,
    ) -> Tuple[Dict[int, int], set]:
        """
        Run Hungarian matching between detections and a pool of tracks.
        Returns (det_idx → track_id, set of matched det indices).
        Distance threshold scales with frame gap.
        """
        from scipy.optimize import linear_sum_assignment

        if not pool or not detections:
            return {}, set()

        tids     = list(pool.keys())
        # Hard cap: dislocations barely move so we don't scale with frame gap.
        # The graveyard handles temporal gaps; spatial tolerance stays fixed.
        inf_dist = max_distance + 1.0
        cost     = np.full((len(detections), len(tids)), inf_dist)

        for i, det in enumerate(detections):
            for j, tid in enumerate(tids):
                cost[i, j] = float(np.hypot(
                    det["centroid"][0] - pool[tid][0],
                    det["centroid"][1] - pool[tid][1],
                ))

        row_ind, col_ind = linear_sum_assignment(cost)
        matches:      Dict[int, int] = {}   # det_idx → track_id
        matched_dets: set            = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_distance:
                matches[r]     = tids[c]
                matched_dets.add(r)

        return matches, matched_dets

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        assigned:     List[Dict]   = []
        matched_tids: set[int]     = set()

        # --- Stage 1: match to active tracks ---
        active_matches, matched_dets = self._match(
            detections, self.active, self.last_seen, frame_idx, self.max_distance
        )
        for det_idx, tid in active_matches.items():
            det = detections[det_idx]
            self.active[tid]    = det["centroid"]
            self.last_seen[tid] = frame_idx
            self.history[tid].append((frame_idx, *det["centroid"]))
            matched_tids.add(tid)
            d = dict(det); d["track_id"] = tid
            assigned.append(d)

        unmatched = [i for i in range(len(detections)) if i not in matched_dets]

        # --- Stage 2: try to revive graveyard tracks ---
        if unmatched and self.graveyard:
            unmatched_dets = [detections[i] for i in unmatched]
            gy_matches, gy_matched = self._match(
                unmatched_dets, self.graveyard, self.graveyard_last_seen,
                frame_idx, self.max_distance
            )
            still_unmatched = []
            for local_idx, orig_idx in enumerate(unmatched):
                if local_idx in gy_matches:
                    tid = gy_matches[local_idx]
                    det = detections[orig_idx]
                    self._revive(tid, det, frame_idx)
                    matched_tids.add(tid)
                    d = dict(det); d["track_id"] = tid
                    assigned.append(d)
                else:
                    still_unmatched.append(orig_idx)
            unmatched = still_unmatched

        # --- Stage 3: new tracks for anything still unmatched ---
        for orig_idx in unmatched:
            det = detections[orig_idx]
            d   = dict(det); d["track_id"] = self._register(det, frame_idx)
            assigned.append(d)

        # --- Age out active tracks → graveyard ---
        for sid in set(self.active.keys()) - matched_tids:
            frame_gap = frame_idx - self.last_seen.get(sid, frame_idx)
            if frame_gap > self.max_age:
                self.graveyard[sid]           = self.active.pop(sid)
                self.graveyard_last_seen[sid] = self.last_seen.pop(sid)

        # --- Expire graveyard entries that are too old to revive ---
        expired = [
            tid for tid, ls in self.graveyard_last_seen.items()
            if (frame_idx - ls) > self.graveyard_age
        ]
        for tid in expired:
            self.graveyard.pop(tid, None)
            self.graveyard_last_seen.pop(tid, None)

        return assigned


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_tracked(
    labels: np.ndarray,
    detections: List[Dict],
    tracker: SimpleTracker,
    frame_idx: int,
) -> np.ndarray:
    h, w  = labels.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for det in detections:
        color = _color_from_id(det["track_id"])
        canvas[labels == det["label"]] = color
        cx, cy = det["centroid"]
        cv2.putText(canvas, str(det["track_id"]),
                    (int(cx) - 6, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for tid, positions in tracker.history.items():
        pts = [(int(x), int(y)) for f, x, y in positions if f <= frame_idx]
        if len(pts) < 2:
            continue
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                      isClosed=False, color=_color_from_id(tid),
                      thickness=1, lineType=cv2.LINE_AA)
    return canvas


def _draw_colorized(labels: np.ndarray, detections: List[Dict]) -> np.ndarray:
    h, w   = labels.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for det in detections:
        canvas[labels == det["label"]] = _color_from_id(det["track_id"])
    return canvas


def _draw_overlay(img_gray: np.ndarray, detections: List[Dict]) -> np.ndarray:
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    contours = [det["contour"] for det in detections]
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay


def _loop_stats(frame_idx: int, detections: List[Dict]) -> Dict:
    n = len(detections)
    if n == 0:
        return {"frame": frame_idx, "number_of_loops": 0,
                "average_loop_size": 0.0, "average_diameter": 0.0}
    areas     = np.array([d["area"] for d in detections], dtype=np.float64)
    diameters = np.sqrt(4.0 * areas / pi)
    return {"frame": frame_idx, "number_of_loops": n,
            "average_loop_size": float(areas.mean()),
            "average_diameter":  float(diameters.mean())}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    image_dir: Path,
    out_dir: Path,
    model: ResAttentionUNet,
    device: str,
    threshold: float = 0.5,
    n_passes: int = 1,
    grain_boundaries: Optional[List] = None,
    fg_thresh_rel: float = FG_THRESH_REL,
    max_age: int = MAX_AGE,
    graveyard_age: int = GRAVEYARD_AGE,
) -> None:
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    # Output directories
    masks_dir    = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    tracks_dir   = out_dir / "tracks"
    color_dir    = out_dir / "colorized"
    for d in [masks_dir, overlays_dir, tracks_dir, color_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tracker   = SimpleTracker(max_distance=MAX_DISTANCE, max_age=max_age, graveyard_age=graveyard_age)
    csv_rows:  List[Dict] = []

    loop_csv_path = out_dir / "loop_stats.csv"
    loop_file     = loop_csv_path.open("w", newline="")
    loop_writer   = csv.DictWriter(
        loop_file,
        fieldnames=["frame", "number_of_loops", "average_loop_size", "average_diameter"],
    )
    loop_writer.writeheader()

    print(f"Processing {len(image_paths)} images…")
    for idx, img_path in enumerate(image_paths):
        frame_idx = _extract_frame_index(img_path, idx)
        img_gray  = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"  Skipping unreadable file: {img_path.name}")
            continue

        # 1. Model inference → binary mask
        binary_mask = predict_mask(model, img_gray, device,
                                   threshold=threshold, n_passes=n_passes)

        # 2. Watershed component detection
        detections, labels = _find_components(binary_mask, fg_thresh_rel=fg_thresh_rel)

        # 3. Tracking
        detections = tracker.update(detections, frame_idx)

        stem = img_path.stem

        # 4. Save outputs
        cv2.imwrite(str(masks_dir    / f"{stem}_mask.png"),     binary_mask)
        cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"),  _draw_overlay(img_gray, detections))
        cv2.imwrite(str(tracks_dir   / f"{stem}_tracked.png"),  _draw_tracked(labels, detections, tracker, frame_idx))
        cv2.imwrite(str(color_dir    / f"{stem}_color.png"),    _draw_colorized(labels, detections))

        # 5. Accumulate CSV rows
        for det in detections:
            gb_dist = (min_dist_to_grain_boundaries(det["centroid"], grain_boundaries)
                       if grain_boundaries else float("nan"))
            csv_rows.append({
                "frame":       frame_idx,
                "track_id":    det["track_id"],
                "area":        det["area"],
                "bbox_x":      det["bbox"][0],
                "bbox_y":      det["bbox"][1],
                "bbox_w":      det["bbox"][2],
                "bbox_h":      det["bbox"][3],
                "centroid_x":  det["centroid"][0],
                "centroid_y":  det["centroid"][1],
                "dist_to_gb_px": gb_dist,
            })
        loop_writer.writerow(_loop_stats(frame_idx, detections))
        loop_file.flush()

        n = len(detections)
        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name} — {n} dislocation(s) detected")

    # 6. Write tracks CSV
    tracks_csv = out_dir / "tracks.csv"
    with open(tracks_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "track_id", "area",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "centroid_x", "centroid_y", "dist_to_gb_px",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    loop_file.close()
    print(f"\nDone.")
    print(f"  Masks     → {masks_dir}")
    print(f"  Overlays  → {overlays_dir}")
    print(f"  Tracks    → {tracks_dir}")
    print(f"  tracks.csv    → {tracks_csv}")
    print(f"  loop_stats.csv → {loop_csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained model on a sequence of TEM images, track dislocations, output stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory of TEM images to process (in frame order).")
    parser.add_argument("--out-dir",   type=Path, default=Path("active_learning_data/tracking"),
                        help="Output directory for masks, overlays, tracks, and CSVs.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold (0–1) for dislocation vs background.")
    parser.add_argument("--passes",    type=int,   default=1,
                        help="MC Dropout forward passes per patch (1 = fast, >1 = smoother).")
    parser.add_argument("--gb-path",   type=Path,  default=None,
                        help="Path to grain_boundaries.json for distance-to-boundary stats.")
    parser.add_argument("--fg-thresh", type=float, default=FG_THRESH_REL,
                        help="Watershed foreground threshold (fraction of max distance transform).")
    parser.add_argument("--max-age",      type=int, default=MAX_AGE,
                        help="Frames before an unmatched track is moved to the graveyard.")
    parser.add_argument("--graveyard-age", type=int, default=GRAVEYARD_AGE,
                        help="Frames a dead track stays revivable in the graveyard.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)

    grain_boundaries = []
    gb_path = args.gb_path or (args.image_dir / "grain_boundaries.json")
    if gb_path.exists():
        with open(gb_path) as f:
            data = json.load(f)
            grain_boundaries = [((p[0][0], p[0][1]), (p[1][0], p[1][1])) for p in data]
        print(f"Loaded {len(grain_boundaries)} grain boundary segment(s) from {gb_path}")

    run(
        image_dir        = args.image_dir,
        out_dir          = args.out_dir,
        model            = model,
        device           = device,
        threshold        = args.threshold,
        n_passes         = args.passes,
        grain_boundaries = grain_boundaries,
        fg_thresh_rel    = args.fg_thresh,
        max_age          = args.max_age,
        graveyard_age    = args.graveyard_age,
    )


if __name__ == "__main__":
    main()
