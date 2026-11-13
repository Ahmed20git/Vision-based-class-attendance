"""Build the face gallery from the enrolled students.

PIPELINE STEP 2 of 3 (enrollment -> gallery -> recognition)

1. Student enrollment (the Register Student page in recognition_app.py, or the
   Enrollment notebook) saves a cropped face photo and a row
   (student_id, student_name, face_image) in Enrollment/students.db.
2. This file reads every row, turns each face photo into a 512-number face
   signature ("embedding") with the Facenet512 deep network, and saves all of
   them in enrollment_gallery.pkl:
       {student_id: {"name": student_name, "embedding": [512 numbers]}}
3. recognize_classroom.py compares every face it finds in a photo or video
   against these embeddings.

Run on its own to rebuild the gallery:  ./.venv/bin/python create_enrollment_gallery.py
"""
import sqlite3
import pickle
from pathlib import Path
from deepface import DeepFace

# Where things live: this folder, the Enrollment folder beside it, the
# student database, and the gallery file this script writes.
BASE_DIR = Path(__file__).resolve().parent
ENROLLMENT_DIR = BASE_DIR.parent / "Enrollment"
DATABASE = ENROLLMENT_DIR / "students.db"
GALLERY_FILE = BASE_DIR / "enrollment_gallery.pkl"


def update_gallery():
    """Rebuild enrollment_gallery.pkl from students.db and return the gallery."""
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Database not found: {DATABASE}")

    # Step 1: read every enrolled student from the database.
    connection = sqlite3.connect(DATABASE)

    try:
        students = connection.execute(
            "SELECT student_id, student_name, face_image FROM students"
        ).fetchall()
    finally:
        connection.close()

    # Step 2: compute one face signature (embedding) per student.
    gallery = {}

    for student_id, student_name, image_path in students:
        full_path = ENROLLMENT_DIR / image_path

        if not full_path.is_file():
            print(f"Image missing for student {student_id}: {full_path}")
            continue

        # The saved photo is already cropped to the face at enrollment, so
        # face detection is skipped ("skip") and the photo goes straight into
        # Facenet512, which returns a 512-number embedding.
        result = DeepFace.represent(
            img_path=str(full_path),
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False
        )

        gallery[student_id] = {
            "name": student_name,
            "embedding": result[0]["embedding"]
        }

        print(f"Registered: {student_id} | {student_name}")

    # Step 3: save the completed gallery for recognize_classroom.py to load.
    with open(GALLERY_FILE, "wb") as file:
        pickle.dump(gallery, file)

    print(f"\nTotal enrolled students: {len(gallery)}")
    print(f"Gallery saved: {GALLERY_FILE}")

    return gallery


if __name__ == "__main__":
    update_gallery()
