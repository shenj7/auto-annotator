#!/usr/bin/env python3
import argparse
import csv
from math import pi
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


MIN_AREA = 5
MAX_DISTANCE = 25.0
MIN_ROUNDNESS = 0.6 
BG_DILATE_ITERS = 3
FG_THRESH_REL = 0.35  # relative to max distance transform value


def _color_from_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic bright-ish BGR color for a track id."""
    rng = np.random.RandomState(track_id * 9973 + 17)
    return tuple(int(x) for x in rng.randint(60, 255, size=3))


def _extract_frame_index(path: Path, default_idx: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else default_idx


def _find_components(
    mask: np.ndarray,
) -> Tuple[List[Dict], np.ndarray]:
    min_area = MIN_AREA
    fg = (mask < 128).astype(np.uint8)
    if fg.max() == 0:
        return [], np.zeros_like(fg, dtype=np.int32)

    kernel = np.ones((3, 3), np.uint8)

    opened = (fg > 0).astype(np.uint8) * 255

    # 2) sure background by dilation
    sure_bg = cv2.dilate(opened, kernel, iterations=BG_DILATE_ITERS)

    # 3) sure foreground via distance transform + threshold
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist <= 0:
        return [], np.zeros_like(fg, dtype=np.int32)
    _, sure_fg = cv2.threshold(
        dist, FG_THRESH_REL * max_dist, 255, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8)

    # 4) unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5) markers from sure foreground
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # reserve 0 for unknown
    markers[unknown == 255] = 0

    # 6) watershed
    color = cv2.cvtColor((fg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)  # markers modified in place
    labels = markers
    labels[labels == -1] = 0  # boundary to background so it won't colorize

    comps: List[Dict] = []
    for label_id in np.unique(labels):
        if label_id <= 1:  # background and watershed boundary (-1) skipped
            continue

        mask_label = (labels == label_id).astype(np.uint8)
        area = int(mask_label.sum())
        if area < min_area:
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

        x, y, w, h = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        comps.append(
            {
                "label": int(label_id),
                "centroid": (cx, cy),
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": area,
            }
        )

    return comps, labels


class SimpleTracker:
    def __init__(self, max_distance: float):
        self.max_distance = float(max_distance)
        self.active: Dict[int, Tuple[float, float]] = {}
        self.history: Dict[int, List[Tuple[int, float, float]]] = {}
        self.next_id = 1

    def _register(self, detection: Dict, frame_idx: int) -> int:
        track_id = self.next_id
        self.next_id += 1
        self.active[track_id] = detection["centroid"]
        self.history[track_id] = [(frame_idx, *detection["centroid"])]
        return track_id

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        assigned: List[Dict] = []
        used_tracks: set[int] = set()

        for det in detections:
            best_id = None
            best_dist = self.max_distance + 1.0
            for track_id, last_pt in self.active.items():
                if track_id in used_tracks:
                    continue
                dist = float(
                    np.hypot(det["centroid"][0] - last_pt[0], det["centroid"][1] - last_pt[1])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is None or best_dist > self.max_distance:
                best_id = self._register(det, frame_idx)
                used_tracks.add(best_id)
            else:
                self.active[best_id] = det["centroid"]
                self.history[best_id].append((frame_idx, *det["centroid"]))
                used_tracks.add(best_id)

            det_with_id = dict(det)
            det_with_id["track_id"] = best_id
            assigned.append(det_with_id)

        stale_ids = set(self.active.keys()) - {d["track_id"] for d in assigned}
        for sid in stale_ids:
            self.active.pop(sid, None)

        return assigned


def _draw_tracks(
    labels: np.ndarray,
    detections: List[Dict],
    tracker: SimpleTracker,
    frame_idx: int,
) -> np.ndarray:
    h, w = labels.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for det in detections:
        color = _color_from_id(det["track_id"])
        canvas[labels == det["label"]] = color
        cx, cy = det["centroid"]
        cv2.putText(
            canvas,
            f"{det['track_id']}",
            (int(cx) - 6, int(cy) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for track_id, positions in tracker.history.items():
        pts = [(int(x), int(y)) for f, x, y in positions if f <= frame_idx]
        if len(pts) < 2:
            continue
        cv2.polylines(
            canvas,
            [np.array(pts, dtype=np.int32)],
            isClosed=False,
            color=_color_from_id(track_id),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def _colorize_components(labels: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Assign each detected component a unique color (no labels or paths)."""
    h, w = labels.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for det in detections:
        canvas[labels == det["label"]] = _color_from_id(det["track_id"])
    return canvas


def _summarize_loops(frame_idx: int, detections: List[Dict]) -> Dict[str, float]:
    """Compute loop-level statistics for a frame."""
    num_loops = len(detections)
    if num_loops == 0:
        return {
            "frame": frame_idx,
            "number_of_loops": 0,
            "average_loop_size": 0.0,
            "average_diameter": 0.0,
        }

    areas = np.array([det["area"] for det in detections], dtype=np.float64)
    # Equivalent circle diameter for each detection
    diameters = np.sqrt(4.0 * areas / pi)
    return {
        "frame": frame_idx,
        "number_of_loops": num_loops,
        "average_loop_size": float(areas.mean()),
        "average_diameter": float(diameters.mean()),
    }


def process_masks(
    mask_dir: Path,
    output_dir: Path,
) -> None:
    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    if not mask_paths:
        raise ValueError(f"No *_mask.png files found in {mask_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = output_dir / "tracks"
    color_dir = output_dir / "colorized"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    tracker = SimpleTracker(max_distance=MAX_DISTANCE)
    csv_rows: List[Dict] = []
    loop_csv_path = output_dir / "loop_stats.csv"
    loop_fieldnames = [
        "frame",
        "number_of_loops",
        "average_loop_size",
        "average_diameter",
    ]
    loop_file = loop_csv_path.open("w", newline="")
    loop_writer = csv.DictWriter(loop_file, fieldnames=loop_fieldnames)
    loop_writer.writeheader()

    for idx, mask_path in enumerate(mask_paths):
        frame_idx = _extract_frame_index(mask_path, idx)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        detections, labels = _find_components(mask)
        detections = tracker.update(detections, frame_idx)

        canvas = _draw_tracks(labels, detections, tracker, frame_idx)
        colorized = _colorize_components(labels, detections)
        out_name = f"{mask_path.stem.replace('_mask', '')}_tracked.png"
        cv2.imwrite(str(tracks_dir / out_name), canvas)
        cv2.imwrite(str(color_dir / out_name.replace("_tracked", "_color")), colorized)

        for det in detections:
            csv_rows.append(
                {
                    "frame": frame_idx,
                    "track_id": det["track_id"],
                    "area": det["area"],
                    "bbox_x": det["bbox"][0],
                    "bbox_y": det["bbox"][1],
                    "bbox_w": det["bbox"][2],
                    "bbox_h": det["bbox"][3],
                    "centroid_x": det["centroid"][0],
                    "centroid_y": det["centroid"][1],
                }
            )
        loop_writer.writerow(_summarize_loops(frame_idx, detections))
        loop_file.flush()

    csv_path = output_dir / "tracks.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "track_id",
                "area",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "centroid_x",
                "centroid_y",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    loop_file.close()

    print(f"Wrote {len(mask_paths)} tracked frames to {tracks_dir}")
    print(f"Wrote track table to {csv_path}")
    print(f"Wrote loop statistics to {loop_csv_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_mask_dir = script_dir / "data" / "auto_images" / "masks"
    default_out_dir = default_mask_dir / "tracking"

    parser = argparse.ArgumentParser(description="Colorize and track binary mask frames.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=default_mask_dir,
        help="Directory containing *_mask.png files (default: src/data/auto_images/masks)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory to write colorized frames and tracks.csv (default: src/data/auto_images/tracking)",
    )
    args = parser.parse_args()

    process_masks(
        mask_dir=args.mask_dir,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
