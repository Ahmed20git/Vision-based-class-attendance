"""Step 3 of the evaluation: find the best threshold.

Uses the same checked ground truth as evaluate.py and scores the app at many
thresholds (by default 0.10 to 0.90 in steps of 0.05).

Remember what the threshold means: it is the largest cosine DISTANCE that still
counts as a match. A lower threshold is stricter (fewer strangers accepted,
more students missed); a higher one is looser.

Choosing a threshold on the same faces you then report on makes the result
look better than it really is. For a fair report, pick the threshold with one
set of photos/videos and report evaluate.py on a different set.

Usage (from the Evaluation folder):
    ../FaceRecognition/.venv/bin/python threshold_checker.py
    ../FaceRecognition/.venv/bin/python threshold_checker.py --thresholds 0.3 0.5 0.9
    ../FaceRecognition/.venv/bin/python threshold_checker.py --only video   the video only
    ../FaceRecognition/.venv/bin/python threshold_checker.py --set real_test   the separate test set

Writes results/threshold_results.csv and, if matplotlib is installed,
results/threshold_chart.png.
"""
import argparse
import csv
import re

from eval_common import (SET_HELP, TEST_SET, RESULTS_DIR, ground_truth, load_cache, load_gallery, matches,
                         predict, score)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[round(0.10 + 0.05 * i, 2) for i in range(17)],
                        help="thresholds to try (default 0.10 to 0.90)")
    parser.add_argument("--only", metavar="TEXT",
                        help="only faces whose source file name contains TEXT (video, photos, or e.g. dark)")
    parser.add_argument("--set", metavar="SET", help=SET_HELP)
    args = parser.parse_args()
    if TEST_SET:
        print(f"Using the separate test set in {TEST_SET}/")

    gallery, cache = load_gallery(), load_cache()
    students = [str(s) for s in gallery]
    faces = [(cache[name]["embedding"], truth) for name, truth in ground_truth()
             if name in cache and matches(cache[name]["source"], args.only)]
    if not faces:
        raise SystemExit("No labelled faces found in ground_truth/.")

    # Score the app at every threshold.
    results = []
    for threshold in sorted(args.thresholds):
        pairs = [(truth, predict(embedding, gallery, threshold)[0]) for embedding, truth in faces]
        result = score(pairs, students)
        result["threshold"] = threshold
        results.append(result)

    header = (f"{'Threshold':>9} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>6} "
              f"{'Strangers accepted':>19} {'Students missed':>16}")
    print(f"{len(faces)} labelled faces\n\n{header}")
    for r in results:
        strangers = (f"{r['false_accepts']}/{r['stranger_faces']}" if r["stranger_faces"] else "-")
        print(f"{r['threshold']:>9.2f} {r['accuracy']:>9.1%} {r['macro_precision']:>10.1%} "
              f"{r['macro_recall']:>7.1%} {r['macro_f1']:>6.1%} {strangers:>19} "
              f"{r['missed_as_unknown']:>16}")

    best_accuracy = max(results, key=lambda r: (r["accuracy"], -r["threshold"]))
    best_f1 = max(results, key=lambda r: (r["macro_f1"], -r["threshold"]))
    print(f"\nBest accuracy: threshold {best_accuracy['threshold']:.2f} "
          f"({best_accuracy['accuracy']:.1%})")
    print(f"Best macro F1: threshold {best_f1['threshold']:.2f} ({best_f1['macro_f1']:.1%})")

    RESULTS_DIR.mkdir(exist_ok=True)
    # e.g. evaluation_report_video.txt, so runs with --only do not overwrite each other.
    suffix = "_" + re.sub(r"\W+", "_", args.only.lower()) if args.only else ""
    with open(RESULTS_DIR / f"threshold_results{suffix}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Threshold", "Accuracy", "Macro precision", "Macro recall", "Macro F1",
                         "Strangers accepted", "Stranger faces", "Students missed",
                         "Wrong student", "Enrolled faces"])
        for r in results:
            writer.writerow([r["threshold"], f"{r['accuracy']:.4f}", f"{r['macro_precision']:.4f}",
                             f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}",
                             r["false_accepts"], r["stranger_faces"], r["missed_as_unknown"],
                             r["wrong_student"], r["enrolled_faces"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nSaved threshold_results.csv in {RESULTS_DIR} "
              "(install matplotlib for a chart)")
        return
    x = [r["threshold"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [r["accuracy"] for r in results], marker="o", label="Accuracy")
    ax.plot(x, [r["macro_f1"] for r in results], marker="o", label="Macro F1")
    ax.plot(x, [r["false_accepts"] / r["stranger_faces"] if r["stranger_faces"] else 0
                for r in results], marker="o", label="Strangers accepted (rate)")
    ax.set_xlabel("Threshold (cosine distance)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"threshold_chart{suffix}.png", dpi=150)
    print(f"\nSaved threshold_results.csv and threshold_chart.png in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
