"""Step 1 of the evaluation: run the app's recognition and sort the faces.

Every face the app detects in the test photos / videos is cut out and saved in
ground_truth/<what the app thinks it is>/:
    ground_truth/101/   faces the app named student 101
    ground_truth/102/   ...
    ground_truth/unknown/   faces the app said are not enrolled

Then YOU check every folder and move each face into the folder of the person
it really is (or into not_a_face/ if it is not a face). After that, the
folders are the ground truth (the correct answers), and evaluate.py and
threshold_checker.py compare the app against them.

Usage (from the Evaluation folder, with the app's Python):
    ../FaceRecognition/.venv/bin/python collect_faces.py              all files in input/
    ../FaceRecognition/.venv/bin/python collect_faces.py --every 5    video: one frame every 5 s
    ../FaceRecognition/.venv/bin/python collect_faces.py --camera 30  live camera for 30 s
    ../FaceRecognition/.venv/bin/python collect_faces.py --set real_test
        a separate test set: reads real_test/input/ and writes real_test/ground_truth/,
        so the main ground_truth/ is never touched

Tip: name the input files after their conditions (e.g. 3m_dim_side.jpg,
class_video.mp4). The name is kept in every face file name, and evaluate.py
reports results for each word in it (3m, dim, side, ...).

Files already processed are skipped, so your corrections are never
overwritten. Delete faces_cache.pkl and ground_truth/ to start again.
"""
import argparse
import pickle
import time
from pathlib import Path

import cv2

from eval_common import (SET_HELP, TEST_SET, CACHE_PATH, IGNORED, INPUT_DIR, TRUTH_DIR, UNKNOWN,
                         load_gallery, recognize_classroom)

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
PAD = 0.20  # extra margin around each saved face, so it is easy to recognize by eye


def read_image(path):
    """Read a photo as an OpenCV (BGR) image, including iPhone HEIC photos."""
    image = cv2.imread(str(path))
    if image is not None:
        return image
    import numpy as np
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def save_faces(frame, gallery, source, frame_number, cache):
    """Recognize one picture and save each face into its predicted folder."""
    # The same call the app makes for every photo / frame (detect, embed,
    # compare, decide with THRESHOLD).
    summary = recognize_classroom.recognize_frame(frame.copy(), gallery, frame)
    height, width = frame.shape[:2]
    for face in summary["faces"]:
        x, y, w, h = [int(v) for v in face["bounding_box"]]
        dx, dy = int(w * PAD), int(h * PAD)
        crop = frame[max(0, y - dy):min(height, y + h + dy), max(0, x - dx):min(width, x + w + dx)]
        if crop.size == 0:
            continue
        label = face["student_id"] or UNKNOWN
        name = f"{Path(source).stem}_f{frame_number:05d}_{face['face_number']:02d}.jpg"
        folder = TRUTH_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / name), crop)
        cache[name] = {
            "embedding": face["_embedding"],
            "source": Path(source).name,
            "frame": frame_number,
            "box": [x, y, w, h],
            "collected_as": label,
            "collected_distance": face["distance"],
        }
    return len(summary["faces"])


def collect_video(path, gallery, every_seconds, cache):
    """Sample one frame every every_seconds seconds and save its faces."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * every_seconds))
    frame_number, faces = 0, 0
    while capture.grab():
        if frame_number % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                faces += save_faces(frame, gallery, path.name, frame_number, cache)
                print(f"  frame {frame_number}: {faces} faces so far")
        frame_number += 1
    capture.release()
    return faces


def collect_camera(seconds, gallery, every_seconds, cache):
    """Use the computer's camera (like the Live Camera tab) for `seconds` seconds."""
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise SystemExit("Could not open the camera.")
    source = f"camera_{time.strftime('%Y%m%d_%H%M%S')}"
    start, frame_number, faces = time.time(), 0, 0
    while time.time() - start < seconds:
        ok, frame = capture.read()
        if ok:
            faces += save_faces(frame, gallery, source, frame_number, cache)
            frame_number += 1
            print(f"  camera frame {frame_number}: {faces} faces so far")
        time.sleep(every_seconds)
    capture.release()
    return faces


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="photos/videos (default: everything in input/)")
    parser.add_argument("--every", type=float, default=2.0,
                        help="video/camera: seconds between checked frames (app uses 2)")
    parser.add_argument("--camera", type=float, metavar="SECONDS",
                        help="use the live camera for this many seconds")
    parser.add_argument("--set", metavar="SET", help=SET_HELP)
    args = parser.parse_args()
    if TEST_SET:
        print(f"Using the separate test set in {TEST_SET}/")

    if TEST_SET and not INPUT_DIR.is_dir() and not args.files and not args.camera:
        INPUT_DIR.mkdir(parents=True)
        raise SystemExit(f"Created {INPUT_DIR}. Put the test photos/videos there and run this again.")
    gallery = load_gallery()
    cache = {}
    if CACHE_PATH.is_file():
        with open(CACHE_PATH, "rb") as file:
            cache = pickle.load(file)
    done = {entry["source"] for entry in cache.values()}
    (TRUTH_DIR / UNKNOWN).mkdir(parents=True, exist_ok=True)
    (TRUTH_DIR / IGNORED).mkdir(parents=True, exist_ok=True)
    for student_id in gallery:
        (TRUTH_DIR / str(student_id)).mkdir(parents=True, exist_ok=True)

    try:
        if args.camera:
            count = collect_camera(args.camera, gallery, args.every, cache)
            print(f"Camera: {count} faces saved")
        else:
            files = [Path(f) for f in args.files] or sorted(INPUT_DIR.glob("*"))
            for path in files:
                kind = path.suffix.lower()
                if kind not in IMAGE_TYPES | VIDEO_TYPES:
                    continue
                if path.name in done:
                    print(f"Skipping {path.name} (already collected)")
                    continue
                print(f"Processing {path.name}")
                if kind in VIDEO_TYPES:
                    count = collect_video(path, gallery, args.every, cache)
                else:
                    count = save_faces(read_image(path), gallery, path.name, 0, cache)
                print(f"  {count} faces saved")
    finally:
        # Save progress even if something fails half way.
        with open(CACHE_PATH, "wb") as file:
            pickle.dump(cache, file)

    print(f"\nDone. Now open {TRUTH_DIR} and move every face into the folder of the "
          f"person it really is ({UNKNOWN}/ for anyone not enrolled, {IGNORED}/ for "
          "anything that is not a face). Then run evaluate.py.")


if __name__ == "__main__":
    main()
