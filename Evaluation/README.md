# Evaluation

Measures how well the attendance app recognizes faces (accuracy, precision,
recall, F1) and which threshold works best. Run every command from this
`Evaluation` folder in Terminal.

1. **Add test material.** Put photos and/or videos in `input/`. Name each file
   after its conditions, for example `3m_dim_side.jpg` or `classroom_video.mp4`.
2. **Collect faces.**
   `../FaceRecognition/.venv/bin/python collect_faces.py`
   Every detected face is saved in `ground_truth/<student ID>/` or
   `ground_truth/unknown/`, based on what the app predicted.
   For the live camera, use `--camera 30` (30 seconds) instead.
3. **Check the folders by hand.** Move every wrongly sorted face into the
   folder of the person it really is: `unknown/` for anyone not enrolled,
   `not_a_face/` for anything that is not a face. The folders are now the
   ground truth.
4. **Evaluate.**
   `../FaceRecognition/.venv/bin/python evaluate.py`
   Writes `results/evaluation_report.txt`, `per_face.csv` and
   `confusion_matrix.csv`.
5. **Compare thresholds.**
   `../FaceRecognition/.venv/bin/python threshold_checker.py`
   Writes `results/threshold_results.csv`, plus `threshold_chart.png` if
   matplotlib is installed.

The threshold is a cosine **distance**, so lower is stricter. The app uses 0.30.
For a fair report, choose the threshold with one set of material and report
`evaluate.py` on a different set.

## A separate test set (e.g. real photos)

To test new material without touching the main `input/` and `ground_truth/`,
add `--set NAME` to every command. Everything then lives in the folder `NAME/`:

1. `../FaceRecognition/.venv/bin/python collect_faces.py --set real_test`
   (the first time, this creates `real_test/input/`; put the photos there and run it again)
2. Sort the faces in `real_test/ground_truth/` by hand.
3. `../FaceRecognition/.venv/bin/python evaluate.py --set real_test`
4. `../FaceRecognition/.venv/bin/python threshold_checker.py --set real_test`

Results go to `real_test/results/`.
