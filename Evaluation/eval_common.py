"""Shared helpers for the three evaluation scripts in this folder.

Folder layout (all inside Evaluation/; with --set NAME the same layout is
used inside the folder NAME/ instead, e.g. real_test/input/):
    input/                 put the test photos and videos here
    ground_truth/          collect_faces.py sorts every detected face in here:
        101/, 102/, ...    one folder per enrolled student
        unknown/           everyone who is not enrolled
        not_a_face/        move false detections here (they are ignored)
    faces_cache.pkl        the face signature (embedding) of every saved face
    results/               evaluate.py and threshold_checker.py write here

The face signatures are saved once by collect_faces.py, so evaluate.py and
threshold_checker.py use exactly the faces the app detected, without running
the slow face detector again.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
APP_DIR = PROJECT_DIR / "FaceRecognition"


def _test_set_name():
    """The name given with --set NAME on the command line, or None.

    Every script accepts --set NAME. With it, the script uses a separate test
    set in the folder NAME/ (NAME/input, NAME/ground_truth, NAME/faces_cache.pkl
    and NAME/results), so the main input/ and ground_truth/ are never touched.
    """
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--set" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--set="):
            return arg.split("=", 1)[1]
    return None


TEST_SET = _test_set_name()
SET_DIR = EVAL_DIR / TEST_SET if TEST_SET else EVAL_DIR
INPUT_DIR = SET_DIR / "input"
TRUTH_DIR = SET_DIR / "ground_truth"
CACHE_PATH = SET_DIR / "faces_cache.pkl"
RESULTS_DIR = SET_DIR / "results"
SET_HELP = ("use the separate test set in the folder SET (SET/input, SET/ground_truth, "
            "SET/results) instead of the main input/ and ground_truth/")

UNKNOWN = "unknown"
IGNORED = "not_a_face"

# Use the app's own recognition code and threshold.
sys.path.insert(0, str(APP_DIR))
import recognize_classroom  # noqa: E402


def load_gallery():
    """The enrolled students: {student_id: {"name", "embedding"}}."""
    with open(APP_DIR / "enrollment_gallery.pkl", "rb") as file:
        return pickle.load(file)


def load_cache():
    """{face file name: {"embedding", "source", ...}} saved by collect_faces.py."""
    if not CACHE_PATH.is_file():
        sys.exit("faces_cache.pkl not found. Run collect_faces.py first.")
    with open(CACHE_PATH, "rb") as file:
        return pickle.load(file)


def ground_truth():
    """[(face file name, true label)] from the folder each face is in now.

    The folder name is the correct answer: a student ID, or "unknown".
    Faces in not_a_face/ are left out.
    """
    if not TRUTH_DIR.is_dir():
        sys.exit("ground_truth/ not found. Run collect_faces.py first.")
    labels = []
    for folder in sorted(p for p in TRUTH_DIR.iterdir() if p.is_dir()):
        if folder.name == IGNORED:
            continue
        for image in sorted(folder.glob("*.jpg")):
            labels.append((image.name, folder.name))
    return labels


VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def matches(source, only):
    """For --only: "video" = faces from videos, "photos" = faces from photos,
    anything else = source file names containing that text."""
    if not only:
        return True
    is_video = source.lower().endswith(VIDEO_TYPES)
    if only.lower() == "video":
        return is_video
    if only.lower() == "photos":
        return not is_video
    return only.lower() in source.lower()


def predict(embedding, gallery, threshold):
    """The app's decision for one face: the closest student if the cosine
    distance is at most threshold, otherwise unknown. Returns (label, distance)."""
    best_id, best_distance = UNKNOWN, float("inf")
    for student_id, student in gallery.items():
        distance = recognize_classroom.cosine_distance(embedding, student["embedding"])
        if distance < best_distance:
            best_id, best_distance = str(student_id), distance
    if best_distance > threshold:
        return UNKNOWN, best_distance
    return best_id, best_distance


def score(pairs, students):
    """Accuracy, precision, recall and F1 from (true label, predicted label) pairs.

    Every face is one decision. "Correct" = an enrolled student got their own
    ID, or a non-enrolled person got unknown.
      Precision(S) = faces correctly named S / all faces named S
      Recall(S)    = faces correctly named S / all faces that really are S
      F1(S)        = 2 x precision x recall / (precision + recall)
    The macro average gives every student the same weight. unknown is scored
    the same way, so its recall is how often strangers are rejected.
    """
    total = len(pairs)
    correct = sum(t == p for t, p in pairs)
    per_class = {}
    for label in list(students) + [UNKNOWN]:
        tp = sum(t == label and p == label for t, p in pairs)
        predicted = sum(p == label for _, p in pairs)
        actual = sum(t == label for t, _ in pairs)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "faces": actual}
    scored = [label for label in students if per_class[label]["faces"]]
    n = len(scored) or 1
    strangers = [(t, p) for t, p in pairs if t == UNKNOWN]
    enrolled = [(t, p) for t, p in pairs if t != UNKNOWN]
    return {
        "faces": total,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
        "macro_precision": sum(per_class[s]["precision"] for s in scored) / n,
        "macro_recall": sum(per_class[s]["recall"] for s in scored) / n,
        "macro_f1": sum(per_class[s]["f1"] for s in scored) / n,
        "enrolled_faces": len(enrolled),
        "wrong_student": sum(p not in (t, UNKNOWN) for t, p in enrolled),
        "missed_as_unknown": sum(p == UNKNOWN for _, p in enrolled),
        "stranger_faces": len(strangers),
        "false_accepts": sum(p != UNKNOWN for _, p in strangers),
    }


def unit_vector(values):
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values) + 1e-12)
