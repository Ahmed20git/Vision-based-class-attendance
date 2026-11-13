# Vision-Based Class Attendance

COSC 330 Introduction to AI, Project 3. A system that takes class attendance
automatically by recognizing registered students in classroom photos, videos
and a live camera.

**Pipeline:** RetinaFace face detection → FaceNet512 face signature (512 values,
via DeepFace) → cosine distance to each registered student → closest student if
the distance is ≤ 0.30, otherwise *Unknown* → attendance saved in SQLite.

## Folders

| Path | Contents |
|---|---|
| `FaceRecognition/recognition_app.py` | Gradio app: Attendance Sheet, Take Attendance (Photo, Video, Live Camera), History, Students |
| `FaceRecognition/recognize_classroom.py` | Detection, embeddings, matching, video sampling (one frame every 2 s) and tracking |
| `FaceRecognition/create_enrollment_gallery.py` | Builds `enrollment_gallery.pkl` from the registered photos |
| `FaceRecognition/attendance_store.py` | SQLite attendance database and CSV export |
| `FaceRecognition/attendance_actions.py` | Connects recognition results to attendance records |
| `FaceRecognition/test_attendance.py` | Unit tests |
| `Enrollment/` | First enrollment website (Jupyter notebook, works on a phone through a share link) |
| `Evaluation/` | Scripts used for the report's evaluation (see `Evaluation/README.md`) |

## How to run

```bash
cd FaceRecognition
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python recognition_app.py
```

Open the link printed in the terminal (by default http://127.0.0.1:7860).
Register students in the **Students** tab, then take attendance in
**Take Attendance**. The databases and the gallery are created on first use.

## Privacy

Student photos, `students.db`, `attendance.db`, `enrollment_gallery.pkl`
(face signatures) and the test videos are **not** included in this repository,
because they contain real people's faces.
