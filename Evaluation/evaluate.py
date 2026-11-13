"""Step 2 of the evaluation: score the app against your checked ground truth.

Run collect_faces.py first and correct the folders in ground_truth/. This
script then takes every face, makes the app's decision again (closest
enrolled student if the cosine distance is at most the threshold, otherwise
unknown), and compares it with the folder you put the face in.

Usage (from the Evaluation folder):
    ../FaceRecognition/.venv/bin/python evaluate.py                  app threshold (0.30)
    ../FaceRecognition/.venv/bin/python evaluate.py --threshold 0.35
    ../FaceRecognition/.venv/bin/python evaluate.py --only video      the video only
    ../FaceRecognition/.venv/bin/python evaluate.py --set real_test   the separate test set

Writes to results/:
    evaluation_report.txt  accuracy, precision, recall, F1 (overall, per student,
                           per condition word in the file names)
    per_face.csv           every face: true label, predicted label, distance
    confusion_matrix.csv   rows = true person, columns = what the app said
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

from eval_common import (SET_HELP, TEST_SET, RESULTS_DIR, UNKNOWN, ground_truth, load_cache, load_gallery,
                         matches, predict, recognize_classroom, score)


def report_text(result, title):
    """The scores as text for the report."""
    lines = [f"== {title} ==",
             f"Faces: {result['faces']}   Accuracy: {result['accuracy']:.1%}"]
    if result["enrolled_faces"]:  # macro scores need at least one enrolled student
        lines.append(f"Macro precision: {result['macro_precision']:.1%}   "
                     f"Macro recall: {result['macro_recall']:.1%}   "
                     f"Macro F1: {result['macro_f1']:.1%}")
    for label, m in result["per_class"].items():
        if m["faces"]:
            lines.append(f"  {label:>8}: precision {m['precision']:.1%}  recall {m['recall']:.1%}"
                         f"  F1 {m['f1']:.1%}  ({m['faces']} faces)")
    if result["enrolled_faces"]:
        lines.append(f"Enrolled faces: {result['enrolled_faces']}, given the wrong student: "
                     f"{result['wrong_student']}, missed (said unknown): {result['missed_as_unknown']}")
    if result["stranger_faces"]:
        lines.append(f"Non-enrolled faces: {result['stranger_faces']}, wrongly accepted as a "
                     f"student: {result['false_accepts']} "
                     f"({result['false_accepts'] / result['stranger_faces']:.1%})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=recognize_classroom.THRESHOLD,
                        help="cosine distance threshold (default: the app's 0.30)")
    parser.add_argument("--only", metavar="TEXT",
                        help="only faces whose source file name contains TEXT "
                             "(video, photos, or e.g. dark)")
    parser.add_argument("--set", metavar="SET", help=SET_HELP)
    args = parser.parse_args()
    if TEST_SET:
        print(f"Using the separate test set in {TEST_SET}/")

    gallery, cache = load_gallery(), load_cache()
    students = [str(s) for s in gallery]
    rows, missing = [], 0

    # Step 1: the app's decision for every face you labelled.
    for name, truth in ground_truth():
        entry = cache.get(name)
        if entry is None:
            missing += 1  # a picture that collect_faces.py did not make
            continue
        if not matches(entry["source"], args.only):
            continue
        predicted, distance = predict(entry["embedding"], gallery, args.threshold)
        rows.append({"face": name, "source": entry["source"], "true": truth,
                     "predicted": predicted, "distance": distance})
    if not rows:
        raise SystemExit("No labelled faces found in ground_truth/.")

    # Step 2: scores for all faces, then for each condition word in the
    # source file names (e.g. 3m, dim, side).
    pairs = [(r["true"], r["predicted"]) for r in rows]
    if not args.only:
        title = "All faces"
    elif args.only.lower() in ("video", "photos"):
        title = f"Faces from the {args.only.lower()}"
    else:
        title = f"Faces from files containing '{args.only}'"
    sections = [report_text(score(pairs, students), f"{title} (threshold {args.threshold:.2f})")]
    # One section per video, and one for all photos together.
    videos = sorted({r["source"] for r in rows
                     if r["source"].lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))})
    if len({r["source"] for r in rows}) > 1:  # a lone video would just repeat the totals
        for source in videos:
            sections.append(report_text(score([(r["true"], r["predicted"]) for r in rows
                                               if r["source"] == source], students),
                                        f"Video: {source}"))
    photo_pairs = [(r["true"], r["predicted"]) for r in rows
                   if not r["source"].lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))]
    if photo_pairs and len(photo_pairs) < len(rows):
        sections.append(report_text(score(photo_pairs, students), "All photos"))
    by_word = defaultdict(list)
    for r in rows:
        if r["source"] in videos:
            continue  # conditions come from photo names only
        stem = r["source"].rsplit(".", 1)[0]
        for word in set(stem.split("_")):
            # Skip numbers and person names like student101 / unknown1.
            if word and not word.isdigit() and not re.fullmatch(r"(student|unknown)\d*", word):
                by_word[word].append((r["true"], r["predicted"]))
    if len(by_word) > 1:
        for word in sorted(by_word):
            sections.append(report_text(score(by_word[word], students), f"Condition: {word}"))
    text = "\n\n".join(sections)
    if missing:
        text += f"\n\n({missing} pictures in ground_truth/ were not made by collect_faces.py and were skipped.)"

    # Step 3: save everything.
    RESULTS_DIR.mkdir(exist_ok=True)
    # e.g. evaluation_report_video.txt, so runs with --only do not overwrite each other.
    suffix = "_" + re.sub(r"\W+", "_", args.only.lower()) if args.only else ""
    (RESULTS_DIR / f"evaluation_report{suffix}.txt").write_text(text + "\n", encoding="utf-8")
    with open(RESULTS_DIR / f"per_face{suffix}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Face", "Source", "True", "Predicted", "Distance", "Correct"])
        for r in rows:
            writer.writerow([r["face"], r["source"], r["true"], r["predicted"],
                             f"{r['distance']:.3f}", "yes" if r["true"] == r["predicted"] else "no"])
    labels = students + [UNKNOWN]
    counts = Counter(pairs)
    with open(RESULTS_DIR / f"confusion_matrix{suffix}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true \\ predicted"] + labels)
        for t in labels:
            writer.writerow([t] + [counts[(t, p)] for p in labels])

    print(text)
    print(f"\nSaved evaluation_report.txt, per_face.csv and confusion_matrix.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
