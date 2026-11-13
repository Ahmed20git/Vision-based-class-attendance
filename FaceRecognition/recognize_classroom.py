"""Recognize enrolled students in a classroom photo, camera frame or video.

PIPELINE STEP 3 of 3 (enrollment -> gallery -> recognition)

For every picture the steps are:
  1. DETECT  - RetinaFace finds every face and returns it cropped and aligned.
  2. EMBED   - Facenet512 turns each face into a 512-number signature
               (embedding). Two photos of the same person give similar
               embeddings; different people give different ones.
  3. COMPARE - each embedding is compared with every student in the gallery
               (enrollment_gallery.pkl) using cosine distance:
               0 = identical, about 1 = unrelated.
  4. DECIDE  - the closest student wins if the distance is at most THRESHOLD
               (0.30). Otherwise the face is labelled "Unknown".

For a video (recognize_video) the same steps run on one frame every 2 seconds,
and faces are then followed across frames so that each person is counted once,
and each unknown person is shown once.

Used by recognition_app.py. Can also run on its own:
    ./.venv/bin/python recognize_classroom.py photo.jpg
    ./.venv/bin/python recognize_classroom.py --video class.mp4
"""
import math
from pathlib import Path
import sys
import cv2
import pickle
import numpy as np
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent
# A face counts as a student only when its cosine distance to that student's
# enrollment face is at or below THRESHOLD. 0.30 is DeepFace's published value
# for Facenet512 with cosine distance, and the value used in all our tests.
THRESHOLD = 0.30  # cosine distance; same fixed value as the app


def cosine_distance(a, b):
    """1 - cosine similarity of two embeddings: 0 = same direction (same face)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return 1 - np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def recognize_image(image_path, save_output=True):
    """Recognize everyone in one photo file and return a summary dict.

    With save_output=True a copy with boxes and names drawn on it is saved as
    recognized_classroom.jpg.
    """
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # Load the enrolled students' embeddings (made by create_enrollment_gallery.py).
    with open(BASE_DIR / "enrollment_gallery.pkl", "rb") as file:
        gallery = pickle.load(file)

    # Detect on the already-loaded pixels, not the file path: DeepFace rejects
    # paths with non-English characters (macOS screenshot names contain a
    # special space before "PM").
    summary = recognize_frame(image, gallery, image.copy())

    if save_output:
        output_path = BASE_DIR / "recognized_classroom.jpg"
        cv2.imwrite(str(output_path), image)
        summary["output_path"] = str(output_path)

    return summary


def recognize_frame(image, gallery, face_input):
    """Run steps 1-4 on one picture.

    image      - BGR picture that boxes and names are drawn on.
    gallery    - {student_id: {"name", "embedding"}} from enrollment_gallery.pkl.
    face_input - the picture (or path) the faces are detected in; usually the
                 same pixels as image, kept separate so drawing never affects
                 detection.

    Returns total_faces, recognized_students (each student once),
    unknown_count, and faces (one entry per face with its name, distance and
    bounding box).
    """
    # Step 1 - DETECT: RetinaFace finds the faces. align=True rotates each face
    # so the eyes are level, which makes the embeddings more reliable.
    faces = DeepFace.extract_faces(
        img_path=face_input,
        detector_backend="retinaface",
        enforce_detection=False,
        align=True
    )

    results = []
    recognized_students = []
    unknown_count = 0

    # Step 2 - EMBED: keep real detections, then embed them all in one batched call.
    # This is much faster than one DeepFace.represent call per face.
    kept = [
        (face_number, face)
        for face_number, face in enumerate(faces, start=1)
        if face.get("confidence", 0) > 0
    ]
    crops = [
        cv2.cvtColor((face["face"] * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        for _, face in kept
    ]
    embeddings = []
    if crops:
        representations = DeepFace.represent(
            img_path=crops,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False
        )
        # A batch of one comes back as a flat list of results.
        if len(crops) == 1 and isinstance(representations[0], dict):
            representations = [representations]
        embeddings = [rep[0]["embedding"] for rep in representations]

    for (face_number, face), embedding in zip(kept, embeddings):
        # Where the face is in the picture (top-left corner, width, height).
        area = face["facial_area"]
        x, y = area["x"], area["y"]
        w, h = area["w"], area["h"]

        # Step 3 - COMPARE: find the enrolled student with the smallest
        # cosine distance to this face.
        best_subject = None
        best_distance = float("inf")

        for student_id, student_data in gallery.items():
            distance = cosine_distance(
                embedding,
                student_data["embedding"]
            )

            if distance < best_distance:
                best_distance = distance
                best_subject = student_id

        # Step 4 - DECIDE: too far from every student means not enrolled.
        if best_subject is None or best_distance > THRESHOLD:
            student_id = None
            name = "Unknown"
            unknown_count += 1
            color = (0, 0, 255)
            label = "Unknown"
        else:
            student_id = str(best_subject)
            name = gallery[best_subject]["name"]
            color = (0, 255, 0)
            label = f"{name} ({student_id})"

            # List each recognized student once, even if matched twice.
            if not any(
                s["student_id"] == student_id
                for s in recognized_students
            ):
                recognized_students.append({
                    "student_id": student_id,
                    "name": name
                })

        results.append({
            "face_number": face_number,
            "student_id": student_id,
            "name": name,
            "distance": float(best_distance),
            "bounding_box": [x, y, w, h],
            # Used by recognize_video to group repeats of the same unknown person.
            "_embedding": np.asarray(embedding, dtype=np.float32),
        })

        # Draw a green (student) or red (unknown) box and label on the picture.
        cv2.rectangle(
            image, (x, y), (x + w, y + h), color, 2
        )
        cv2.putText(
            image, label, (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

        print(
            f"Face {face_number}: {label} "
            f"| Distance: {best_distance:.4f}"
        )

    return {
        "total_faces": len(results),
        "recognized_students": recognized_students,
        "unknown_count": unknown_count,
        "faces": results,
        "output_path": None
    }


# ---------------------------------------------------------------------------
# VIDEO: following faces across frames
# ---------------------------------------------------------------------------
# A video shows the same people in many frames. To count each person once:
#   1. track_faces   - link each face to the face in the same seat in the
#                      previous sampled frame (the classroom camera is fixed).
#   2. unknown_groups - decide which tracks are students and which are
#                      unknown, and join tracks that belong to the same
#                      unknown person.
#   3. unknown_thumbnails - cut out one clear picture per unknown person.
#
# Grouping faces across video frames (tuned on the instructor's 2-minute
# classroom video against its answer key: 16 of 17 labelled strangers found,
# no one listed twice).
SAME_SEAT_IOU = 0.30           # same place in the next sampled frame = same person
SAME_SEAT_DISTANCE = 0.60      # face centres closer than 0.6 face-widths = same seat
SAME_PERSON_SIMILARITY = 0.60  # join two groups whose average faces are this similar
STUDENT_SHARE = 0.25           # a group is a student if >= 25% of its faces matched them
MIN_FRAMES_FOR_UNKNOWN = 2     # ignore strangers seen in only one sampled frame
MAX_UNKNOWN_FACES = 100


def box_iou(a, b):
    """Overlap of two [x, y, w, h] boxes: 0 = none, 1 = the same box."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def track_faces(frame, summary, tracks, frame_index):
    """Follow each person from one sampled frame to the next.

    The classroom camera does not move, so a face in the same place as a face
    in the previous sampled frame is the same person. Each track remembers
    how often it matched each student and where (and how) it was seen.
    """
    used = set()
    faces = sorted(summary.get("faces", []),
                   key=lambda f: -f["bounding_box"][2] * f["bounding_box"][3])
    for face in faces:
        if "_embedding" not in face:
            continue
        box = [int(v) for v in face["bounding_box"]]
        emb = face["_embedding"] / (np.linalg.norm(face["_embedding"]) + 1e-12)

        match, best = None, 0.0
        for i, track in enumerate(tracks):
            if i in used:
                continue  # two faces in one frame are two different people
            overlap = box_iou(track["box"], box)
            if overlap >= SAME_SEAT_IOU and overlap > best:
                match, best = i, overlap
        if match is None:
            tracks.append({"box": box, "sum": np.zeros_like(emb), "faces": 0,
                           "frames": set(), "seen": [], "students": {}})
            match = len(tracks) - 1
        used.add(match)

        track = tracks[match]
        track["box"] = box
        track["sum"] = track["sum"] + emb
        track["faces"] += 1
        track["frames"].add(frame_index)
        track["seen"].append((frame_index, box, emb.astype(np.float16)))
        sid = face.get("student_id")
        if sid is not None:
            track["students"][sid] = track["students"].get(sid, 0) + 1


def _near(seen_a, seen_b):
    """True when two tracks were ever in the same seat (centres close)."""
    for _, (ax, ay, aw, ah), _ in seen_a[-5:] + seen_a[:5]:
        for _, (bx, by, bw, bh), _ in seen_b[-5:] + seen_b[:5]:
            size = (aw + bw) / 2
            if abs((ax + aw / 2) - (bx + bw / 2)) < SAME_SEAT_DISTANCE * size and \
               abs((ay + ah / 2) - (by + bh / 2)) < SAME_SEAT_DISTANCE * size:
                return True
    return False


def _student_for_average(face_sum, gallery):
    """The student whose enrollment face is within THRESHOLD of this average face.

    Averaging many sightings of one person removes much of the noise from
    pose, blur and lighting, so a student who rarely matched in single
    frames can still match on their average face.
    """
    best_id, best_distance = None, float("inf")
    for student_id, student_data in (gallery or {}).items():
        distance = cosine_distance(face_sum, student_data["embedding"])
        if distance < best_distance:
            best_id, best_distance = student_id, distance
    return str(best_id) if best_id is not None and best_distance <= THRESHOLD else None


def unknown_groups(tracks, gallery=None, found_students=None):
    """Group the unknown tracks so each unknown person appears once.

    With a gallery, a track or group whose average face matches a student is
    treated as that student, not as unknown, and the student is added to
    found_students (a dict of student_id -> {"student_id", "name"}).
    """
    def is_student(face_sum):
        student_id = _student_for_average(face_sum, gallery)
        if student_id is not None and found_students is not None:
            found_students.setdefault(student_id, {
                "student_id": student_id, "name": gallery[student_id]["name"]})
        return student_id is not None

    # Keep only the tracks that are not students. A track is a student when
    # at least STUDENT_SHARE of its faces matched one student, or when its
    # average face matches a student.
    groups = []
    for track in tracks:
        if track["students"]:
            top = max(track["students"].values())
            if top >= max(1, STUDENT_SHARE * track["faces"]):
                continue  # this is an enrolled student
        if is_student(track["sum"]):
            continue
        groups.append({"sum": track["sum"].copy(), "frames": set(track["frames"]),
                       "seen": list(track["seen"])})

    # Join groups of the same stranger. Groups seen in the same frame are
    # always different people. Otherwise join when they sat in the same seat
    # at different times (a broken track, e.g. the person looked down), or
    # when their average faces are similar (they moved seats).
    while True:
        means = [g["sum"] / (np.linalg.norm(g["sum"]) + 1e-12) for g in groups]
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i]["frames"] & groups[j]["frames"]:
                    continue
                similarity = float(np.dot(means[i], means[j]))
                score = 2.0 if _near(groups[i]["seen"], groups[j]["seen"]) else similarity
                if score >= SAME_PERSON_SIMILARITY and (best is None or score > best[0]):
                    best = (score, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = groups[i], groups.pop(j)
        a["sum"] = a["sum"] + b["sum"]
        a["frames"] |= b["frames"]
        a["seen"] += b["seen"]

    # Drop people seen only once (often a passer-by or a false detection), and
    # check the joined groups against the gallery once more.
    groups = [g for g in groups
              if len(g["frames"]) >= MIN_FRAMES_FOR_UNKNOWN and not is_student(g["sum"])]
    return groups[:MAX_UNKNOWN_FACES]


def best_sighting(group):
    """The sighting to use as the thumbnail: among the sightings that look
    most like the person's average face (so a passer-by or a blocked face is
    not chosen), take the largest."""
    mean = group["sum"] / (np.linalg.norm(group["sum"]) + 1e-12)
    scored = sorted(group["seen"],
                    key=lambda s: -float(np.dot(s[2].astype(np.float32), mean)))
    top = scored[:max(1, len(scored) // 4)]
    frame_index, box, _ = max(top, key=lambda s: s[1][2] * s[1][3])
    return frame_index, box


def unknown_thumbnails(video_path, groups, pad=0.25):
    """Read the chosen frames again and cut out one face per unknown person."""
    from PIL import Image  # Pillow comes with Gradio

    picks = [best_sighting(g) for g in groups]
    crops = [None] * len(picks)
    capture = cv2.VideoCapture(str(video_path))
    try:
        for n in sorted(range(len(picks)), key=lambda k: picks[k][0]):
            frame_index, (x, y, w, h) = picks[n]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            dx, dy = int(w * pad), int(h * pad)
            x1, y1 = max(0, x - dx), max(0, y - dy)
            x2, y2 = min(width, x + w + dx), min(height, y + h + dy)
            if x2 - x1 > 4 and y2 - y1 > 4:
                crops[n] = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return [c for c in crops if c is not None]


def recognize_video(video_path, sample_interval_seconds=2.0, stop_when_all_found=False):
    """Sample a recording and return each recognized student once.

    One frame every 2 seconds is enough for attendance: a student who is in
    class appears in many frames. When every enrolled student has already been
    recognized, the rest of the video cannot add a student, so passing
    stop_when_all_found=True stops early (off by default so the whole video,
    including every unknown person, is checked).
    """
    # Check the input, then load the enrolled students.
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("The sampling interval must be a positive number.")

    video_path = BASE_DIR / Path(video_path)
    with open(BASE_DIR / "enrollment_gallery.pkl", "rb") as file:
        gallery = pickle.load(file)

    capture = cv2.VideoCapture(str(video_path))
    students = {}
    total_faces = 0
    unknown_count = 0
    frames_sampled = 0
    stopped_early = False
    tracks = []  # everyone seen, followed across sampled frames

    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Could not read the video frame rate: {video_path}")

        # Only every frame_step-th frame is analysed (e.g. 30 fps x 2 s = 60).
        frame_step = max(1, round(fps * sample_interval_seconds))
        frame_number = 0
        while capture.grab():
            if frame_number % frame_step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f"Could not decode video frame {frame_number}.")

                # Run steps 1-4 on this frame, add its students to the list,
                # and follow its faces into the next frame.
                # Draw on a copy so unknown-face crops stay clean.
                summary = recognize_frame(frame.copy(), gallery, frame)
                frames_sampled += 1
                total_faces += summary["total_faces"]
                unknown_count += summary["unknown_count"]
                for student in summary["recognized_students"]:
                    students[student["student_id"]] = student
                track_faces(frame, summary, tracks, frame_number)

                if stop_when_all_found and gallery and len(students) >= len(gallery):
                    stopped_early = True
                    break

            frame_number += 1
    finally:
        capture.release()

    if frames_sampled == 0:
        raise ValueError(f"No readable frames in video: {video_path}")

    # Students recognized only on their average face are added here too.
    groups = unknown_groups(tracks, gallery, students)
    return {
        "frames_sampled": frames_sampled,
        "total_faces": total_faces,
        "unknown_count": unknown_count,
        "recognized_students": list(students.values()),
        "stopped_early": stopped_early,
        # One crop per unknown person (repeats across frames are merged).
        # PIL RGB images, ready for attendance_store.save_unknown_faces().
        "unknown_faces": unknown_thumbnails(video_path, groups),
    }


# Command-line use: a photo path, or --video and a video path.
if __name__ == "__main__":
    image_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "classroom_test.jpg"
    )

    if len(sys.argv) == 3 and sys.argv[1] == "--video":
        summary = recognize_video(sys.argv[2])
        print("Sampled frames:", summary["frames_sampled"])
        print("Unique recognized students:", len(summary["recognized_students"]))
        for student in summary["recognized_students"]:
            print(f'{student["student_id"]} | {student["name"]}')
        sys.exit(0)

    if len(sys.argv) > 2 or image_path == "--video":
        sys.exit("Usage: recognize_classroom.py [photo | --video recording]")

    summary = recognize_image(image_path)

    print("\n========== CLASSROOM RESULTS ==========")
    print("Total face detections:", summary["total_faces"])
    print("Unique recognized students:",
          len(summary["recognized_students"]))
    print("Unknown detections:", summary["unknown_count"])
    print("Saved result to:", summary["output_path"])
