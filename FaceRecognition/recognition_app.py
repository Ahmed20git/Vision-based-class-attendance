"""Local Gradio app for the Vision-Based Class Attendance project.

Run from the FaceRecognition folder:
    ./.venv/bin/python recognition_app.py

Tabs:
    Attendance Sheet     - every enrolled student for the chosen class: Present in
                           green, Absent in red. Export it as CSV.
    Take Attendance      - Photo, Video or Live Camera. Recognized students are
                           marked Present; unknown faces are shown in red.
    History              - all saved classes with Present and Absent students.
    Students             - register a student (same steps as the enrollment
                           notebook) and list registered students.

Attendance storage and CSV export are Student 3's code (attendance_store.py and
attendance_actions.py). Records are saved in ../Attendance/attendance.db.

How the files fit together:
    recognition_app.py          this file: the web page (Gradio) and its buttons
    create_enrollment_gallery.py turns enrolled face photos into embeddings
                                 (enrollment_gallery.pkl)
    recognize_classroom.py      detects faces, embeds them with Facenet512 and
                                 compares them to the gallery (cosine distance)
    attendance_actions.py       turns recognition results into attendance and
                                 builds the tables shown on the page
    attendance_store.py         saves attendance, unknown faces and CSV exports
    test_attendance.py          automated tests for the attendance code

What happens when you press a button:
    Register Student -> crop_face() finds the one face in the photo -> the crop
        is saved in Enrollment/registered_faces and a row added to students.db
        -> the gallery is rebuilt so the new student can be recognized.
    Analyze Photo / Video / Live Camera -> recognize_classroom finds and names
        the faces -> attendance_actions saves Present students and unknown faces
        -> the attendance sheet on the page is refreshed.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
import time
import shutil
import sqlite3
import sys
import tempfile
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np
import pandas as pd  # installed with Gradio
from PIL import Image, ImageOps

# iPhone photos are often HEIC, sometimes even when the name ends in .jpg.
# OpenCV can't read HEIC, so convert them when pillow-heif is installed:
#     ./.venv/bin/python -m pip install pillow-heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# Make sure recognize_classroom.py (in this same folder) can be imported
# even if the app is launched from another working directory.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from recognize_classroom import (  # noqa: E402
    BASE_DIR, recognize_frame, recognize_image, recognize_video,
)
import recognize_classroom  # noqa: E402
import attendance_actions  # noqa: E402
import attendance_store  # noqa: E402

# ---------------------------------------------------------------------------
# Recognition threshold (fixed)
# ---------------------------------------------------------------------------
# A face only counts as a student when its cosine distance to that student's
# enrolled face is at or below this value. 0.30 is DeepFace's published value
# for Facenet512 with cosine distance. On our own photos (2026-11-06) correct
# matches were 0.02-0.27, a look-alike relative 0.36, and strangers 0.63 or
# more. All our experiments use 0.30, so it is fixed here, not adjustable.
THRESHOLD = 0.30
recognize_classroom.THRESHOLD = THRESHOLD


def current_threshold():
    """The fixed recognition threshold (cosine distance)."""
    return THRESHOLD


from create_enrollment_gallery import DATABASE, ENROLLMENT_DIR, update_gallery  # noqa: E402
from deepface import DeepFace  # noqa: E402

GALLERY_PATH = BASE_DIR / "enrollment_gallery.pkl"
GALLERY_BACKUP_PATH = BASE_DIR / "enrollment_gallery.backup.pkl"
FACE_FOLDER = ENROLLMENT_DIR / "registered_faces"
TABLE_HEADERS = ["Student ID", "Name"]


# ---------------------------------------------------------------------------
# Small helpers shared by both tabs
# ---------------------------------------------------------------------------

def gallery_missing_message():
    """Warning text if enrollment_gallery.pkl does not exist yet, else None."""
    if GALLERY_PATH.exists():
        return None
    return (
        "⚠️ **Enrollment gallery not found.**\n\n"
        f"Expected `{GALLERY_PATH.name}` next to `recognize_classroom.py` "
        f"(`{GALLERY_PATH}`). Register a student or click **Rebuild Gallery** "
        "in the Registered Students tab, then try again."
    )


# ---------------------------------------------------------------------------
# Enrollment (same logic as the enrollment notebook)
# ---------------------------------------------------------------------------

def ensure_database():
    """Create the registered_faces folder and the students table if missing."""
    FACE_FOLDER.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                student_name TEXT,
                face_image TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def detect_faces(bgr):
    """RetinaFace (the same detector recognition uses). Returns real detections only."""
    faces = DeepFace.extract_faces(
        img_path=bgr,
        detector_backend="retinaface",
        enforce_detection=False,
        align=True,
    )
    # With enforce_detection=False DeepFace returns the whole image with
    # confidence 0 when it finds nothing, so drop that placeholder.
    return [f for f in faces if f.get("confidence", 0) > 0]


def crop_face(photo):
    """Return the single face in a PIL photo, cropped for enrollment."""
    image = photo.convert("RGB")
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    try:
        faces = detect_faces(bgr)
        if not faces:
            # A photo that is already a tight face crop (forehead or chin cut
            # off, e.g. a still from the classroom video) can be missed. Add a
            # border and look again; if exactly one face is there, the photo
            # itself is the face crop.
            pad = max(bgr.shape[:2]) // 3
            padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            if len(detect_faces(padded)) == 1:
                return image
    except Exception as error:
        raise ValueError(f"Face detection failed: {error}") from error

    if not faces:
        raise ValueError("No face was found. Please use a clear, front-facing photo.")
    if len(faces) != 1:
        raise ValueError(
            f"Found {len(faces)} faces. Please use a photo with exactly one face."
        )

    area = faces[0]["facial_area"]
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    return image.crop((max(0, x), max(0, y), min(image.width, x + w), min(image.height, y + h)))


def rebuild_gallery():
    """Back up the current gallery, then rebuild it from students.db."""
    if GALLERY_PATH.exists():
        shutil.copy2(GALLERY_PATH, GALLERY_BACKUP_PATH)
    return update_gallery()


def gallery_summary(gallery):
    """Describe the rebuilt gallery and flag students whose photo was missing."""
    registered = load_registered_students()["Student ID"].tolist()
    skipped = [sid for sid in registered if sid not in gallery]
    message = f"Gallery rebuilt with {len(gallery)} of {len(registered)} registered students."
    if skipped:
        message += (
            "\n\n⚠️ Skipped because their face image is missing: "
            + ", ".join(skipped)
        )
    return message


def load_registered_students():
    """Table of every registered student (ID and name) for the Students tab."""
    ensure_database()
    conn = sqlite3.connect(DATABASE)
    try:
        rows = conn.execute(
            "SELECT student_id, student_name FROM students ORDER BY student_id"
        ).fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=TABLE_HEADERS)


def register_student(student_id, student_name, photo):
    """Enroll one student (the Register Student button).

    Steps: check the inputs -> refuse a duplicate ID -> crop the single face ->
    save the crop and a students.db row -> rebuild the gallery.
    Returns the status message and the saved face crop.
    """
    # Step 1: check that an ID, a name and a photo were given.
    student_id = (student_id or "").strip()
    student_name = (student_name or "").strip()

    if not student_id or not student_name:
        return "⚠️ Please enter both a Student ID and a Full Name.", None
    if photo is None:
        return "⚠️ Please upload or take a face photo.", None

    ensure_database()
    conn = sqlite3.connect(DATABASE)
    try:
        # Step 2: each Student ID can only be registered once.
        existing = conn.execute(
            "SELECT student_name FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if existing:
            return (
                f"⚠️ Student ID **{student_id}** is already registered "
                f"as {existing[0]}.",
                None,
            )

        # Step 3: find exactly one face in the photo and crop it.
        try:
            face = crop_face(photo)
        except ValueError as error:
            return f"⚠️ {error}", None

        # Step 4: save the cropped face and add the student to students.db.
        # Store the path relative to the Enrollment folder, like the
        # existing rows, so the project folder can be moved or shared.
        relative_path = f"registered_faces/{uuid4().hex}.jpg"
        image_path = ENROLLMENT_DIR / relative_path
        face.save(image_path)

        try:
            conn.execute(
                "INSERT INTO students VALUES (?, ?, ?)",
                (student_id, student_name, relative_path),
            )
            conn.commit()
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
    finally:
        conn.close()

    # Step 5: recompute the gallery embeddings so the new student is recognized.
    try:
        gallery = rebuild_gallery()
    except Exception as error:
        return (
            f"⚠️ **{student_name}** was registered, but the gallery rebuild "
            f"failed, so they won't be recognized yet.\n\nDetails: `{error}`\n\n"
            "Try **Rebuild Gallery** in the Registered Students tab.",
            face,
        )

    return (
        f"✅ **{student_name}** ({student_id}) registered.\n\n"
        + gallery_summary(gallery),
        face,
    )


def rebuild_gallery_clicked():
    """The Rebuild Gallery button."""
    try:
        gallery = rebuild_gallery()
    except Exception as error:
        return f"⚠️ **Gallery rebuild failed.**\n\nDetails: `{error}`"
    return "✅ " + gallery_summary(gallery)


# ---------------------------------------------------------------------------
# Attendance helpers (Student 3's attendance_store / attendance_actions)
# ---------------------------------------------------------------------------

def today():
    """Today's date in UAE time (UTC+4), used as the default class date."""
    return datetime.now(timezone(timedelta(hours=4))).date().isoformat()


def clean_date(class_date):
    """The date picker can send '2026-11-06' or '2026-11-06 00:00:00'."""
    return (class_date or "").strip()[:10]


HEIC_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1"}


def is_heic(path):
    """True for an iPhone HEIC file, checked from its first bytes, not its name."""
    with open(path, "rb") as file:
        head = file.read(12)
    return head[4:8] == b"ftyp" and head[8:12] in HEIC_BRANDS


def photo_as_jpeg(photo_path):
    """Return a path OpenCV can read, converting an iPhone HEIC photo to JPEG."""
    if not is_heic(photo_path):
        return photo_path
    if not HEIC_SUPPORT:
        raise ValueError(
            "This is an iPhone HEIC photo (even if its name ends in .jpg). "
            "Export it as JPEG from Preview (File > Export), or install HEIC "
            "support once with: ./.venv/bin/python -m pip install pillow-heif"
        )
    with Image.open(photo_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    jpeg_path = Path(tempfile.gettempdir()) / f"attendance_{uuid4().hex}.jpg"
    image.save(jpeg_path, "JPEG", quality=92)
    return str(jpeg_path)


def check_inputs(media, kind, session, class_date):
    """Return a warning message, or None if everything needed is there."""
    if not media:
        return f"⚠️ Please upload a classroom {kind} first."
    problem = gallery_missing_message()
    if problem:
        return problem
    try:
        attendance_store.validate_session(session, class_date)
    except ValueError as error:
        return f"⚠️ {error}"
    return None


def show_sheet(session, class_date):
    """Attendance sheet for the class chosen at the top of the page."""
    return attendance_actions.roster_html(session, clean_date(class_date))


def class_unknowns(session, class_date):
    """Unknown face pictures saved for the chosen class (newest first)."""
    try:
        faces = attendance_store.unknown_faces(session, clean_date(class_date))
    except ValueError:
        faces = []
    return attendance_actions.unknown_faces_html(faces, "Unknown faces in this class")


def session_choices(current=None):
    """Fill the Class / session list with the sessions that already have records."""
    choices = attendance_store.sessions()
    if current and current not in choices:
        choices = [current] + choices
    return gr.Dropdown(choices=choices, value=current)


def student_choices():
    """Fill the Student list used by Mark Present / Mark Absent."""
    try:
        enrolled = attendance_store.registered_students()
    except FileNotFoundError:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=[(f"{sid} · {name}", sid) for sid, name in
                                sorted(enrolled.items(), key=lambda kv: attendance_store._id_order(kv[0]))],
                       value=None)


def change_status(student_id, present, session, class_date):
    """Mark Present / Mark Absent buttons: manual correction of one student."""
    class_date = clean_date(class_date)
    if not student_id:
        return "⚠️ Choose a student first.", show_sheet(session, class_date)
    try:
        attendance_store.set_status(student_id, present, session, class_date)
    except ValueError as error:
        return f"⚠️ {error}", show_sheet(session, class_date)
    word = "Present" if present else "Absent"
    return f"✅ Student {student_id} marked {word}.", show_sheet(session, class_date)


def clear_this_class(confirmed, session, class_date):
    """Clear this class button (needs the confirmation box ticked)."""
    class_date = clean_date(class_date)
    if not confirmed:
        return "⚠️ Tick the confirmation box first.", show_sheet(session, class_date), \
            class_unknowns(session, class_date), False
    try:
        removed = attendance_store.clear_class(session, class_date)
    except ValueError as error:
        return f"⚠️ {error}", show_sheet(session, class_date), class_unknowns(session, class_date), False
    return (f"🗑️ Cleared {removed} attendance record(s) and the unknown faces for "
            f"{session} on {class_date}."), show_sheet(session, class_date), class_unknowns(session, class_date), False


def delete_everything(confirmed, session, class_date):
    """Delete all history button (needs the confirmation box ticked)."""
    class_date = clean_date(class_date)
    if not confirmed:
        return "⚠️ Tick the confirmation box first.", show_sheet(session, class_date), \
            class_unknowns(session, class_date), False
    removed = attendance_store.delete_all()
    return (f"🗑️ Deleted all attendance history ({removed} record(s)) and all unknown faces. "
            "Enrolled students were kept."), show_sheet(session, class_date), class_unknowns(session, class_date), False


def export_sheet(session, class_date):
    """Export CSV button on the Attendance Sheet tab."""
    try:
        path = attendance_store.export_csv(session, clean_date(class_date))
        return gr.File(value=path, visible=True), "✅ CSV ready below."
    except Exception as error:
        return gr.File(value=None, visible=False), f"⚠️ {error}"


# ---------------------------------------------------------------------------
# Photo handler
# ---------------------------------------------------------------------------

def analyze_photo(photo_path, session, class_date):
    """Analyze Photo button: check inputs, convert HEIC, then take attendance."""
    class_date = clean_date(class_date)
    problem = check_inputs(photo_path, "photo", session, class_date)
    if problem:
        return problem, None, "", show_sheet(session, class_date), ""

    try:
        photo_path = photo_as_jpeg(photo_path)
    except ValueError as error:
        return f"⚠️ **Could not read this photo.** {error}", None, "", \
            show_sheet(session, class_date), ""

    try:
        # Recognition -> recognized_students -> attendance_store.mark_attendance
        return attendance_actions.process_photo(
            photo_path, session, class_date, recognize_image, current_threshold()
        )
    except FileNotFoundError as error:
        message = ("⚠️ **Could not read this photo.** Try a JPG or PNG file.\n\n"
                   f"Details: `{error}`")
    except Exception as error:  # keep the app running and show what failed
        message = f"⚠️ **Photo analysis failed.**\n\nDetails: `{error}`"
    return message, None, "", show_sheet(session, class_date), ""


# ---------------------------------------------------------------------------
# Video handler
# ---------------------------------------------------------------------------

def analyze_video(video_path, session, class_date):
    """Analyze Video button: check inputs, then take attendance from the video."""
    class_date = clean_date(class_date)
    problem = check_inputs(video_path, "video", session, class_date)
    if problem:
        return problem, show_sheet(session, class_date), ""

    try:
        # Uses recognize_video's defaults: one frame every 2 seconds through
        # the whole video.
        return attendance_actions.process_video(
            video_path, session, class_date, recognize_video
        )
    except (ValueError, FileNotFoundError) as error:
        message = ("⚠️ **Could not read this video.** Try an MP4 or MOV recording.\n\n"
                   f"Details: `{error}`")
    except Exception as error:  # keep the app running and show what failed
        message = f"⚠️ **Video analysis failed.**\n\nDetails: `{error}`"
    return message, show_sheet(session, class_date), ""


# ---------------------------------------------------------------------------
# Live camera handler
# ---------------------------------------------------------------------------

LIVE_SECONDS_BETWEEN_FRAMES = 2  # recognition takes ~1-2 s per frame on a laptop

# Browser script: Gradio's streaming webcam needs a "Record" click before it
# sends frames (nothing is recorded or saved). This starts live attendance as
# soon as the Live Camera tab is open, stops it when you leave the tab, and
# renames Gradio's Record / Stop buttons.
LIVE_CAMERA_JS = """
() => {
  const press = (b) => ["pointerdown", "mousedown", "pointerup", "mouseup", "click"]
    .forEach((t) => b.dispatchEvent(new MouseEvent(t, {bubbles: true, cancelable: true, view: window})));
  const relabel = (b, from, to) => {
    const walker = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const n = walker.currentNode;
      if (n.textContent.trim() === from) n.textContent = to;
    }
  };
  let lastPress = 0;
  setInterval(() => {
    const cam = document.getElementById("live-cam");
    if (!cam) return;
    const buttons = [...cam.querySelectorAll("button")];
    const find = (re) => buttons.find((b) => re.test(b.textContent.trim()));
    const start = find(/^(Record|Start live attendance)$/);
    const stop = find(/^(Stop|Pause live attendance)$/);
    if (start) relabel(start, "Record", "Start live attendance");
    if (stop) relabel(stop, "Stop", "Pause live attendance");
    const visible = cam.offsetParent !== null;
    if (Date.now() - lastPress < 2500) return;
    if (visible) {
      const access = find(/Access Webcam/);
      const target = access || (!window.__livePaused && start);
      if (target) { press(target); lastPress = Date.now(); }
    } else if (stop) {
      press(stop); lastPress = Date.now();
    }
  }, 800);
  // A click on "Pause live attendance" by the user keeps it paused.
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest("#live-cam button");
    if (!b || !e.isTrusted) return;
    const text = b.textContent.trim();
    if (/Pause live attendance|Stop/.test(text)) window.__livePaused = true;
    if (/Start live attendance|Record/.test(text)) window.__livePaused = false;
  }, true);
}
"""
_gallery_cache = {"mtime": None, "gallery": None}


def load_gallery():
    """Load enrollment_gallery.pkl once, and again only if it changes."""
    mtime = GALLERY_PATH.stat().st_mtime
    if _gallery_cache["mtime"] != mtime:
        with open(GALLERY_PATH, "rb") as file:
            _gallery_cache["gallery"] = pickle.load(file)
        _gallery_cache["mtime"] = mtime
    return _gallery_cache["gallery"]


LIVE_UNKNOWN_EVERY_SECONDS = 10  # save at most one batch of unknown faces per 10 s
_last_unknown_save = {}


def live_frame(frame, session, class_date):
    """Runs on one webcam frame every LIVE_SECONDS_BETWEEN_FRAMES seconds."""
    class_date = clean_date(class_date)
    if frame is None:
        return None, "Waiting for the camera…", show_sheet(session, class_date), \
            class_unknowns(session, class_date)

    problem = check_inputs(True, "", session, class_date)
    if problem:
        return frame, problem, show_sheet(session, class_date), ""

    try:
        # Recognize the faces in this frame, apply the threshold rules, and
        # mark the recognized students Present.
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        summary = recognize_frame(image.copy(), load_gallery(), image)
        summary = attendance_actions.refine(summary, current_threshold())
        message = attendance_actions.record(summary, session, class_date, "live")
        # Save unknown faces at most once every LIVE_UNKNOWN_EVERY_SECONDS,
        # so one stranger sitting in front of the camera is not saved 100 times.
        picture = Image.fromarray(frame)
        key = (session, class_date)
        if summary["unknown_count"] and \
                time.time() - _last_unknown_save.get(key, 0) >= LIVE_UNKNOWN_EVERY_SECONDS:
            crops = attendance_actions.unknown_crops(picture, summary)
            if attendance_store.save_unknown_faces(crops, session, class_date, "Live"):
                _last_unknown_save[key] = time.time()
        annotated = attendance_actions.annotate_photo(None, summary, picture)
    except Exception as error:  # keep streaming even if one frame fails
        return frame, f"⚠️ **Live recognition failed on this frame.**\n\nDetails: `{error}`", \
            show_sheet(session, class_date), class_unknowns(session, class_date)

    sheet = attendance_actions.roster_html(session, class_date, summary["unknown_count"])
    return annotated, message, sheet, class_unknowns(session, class_date)


# ---------------------------------------------------------------------------
# History handlers
# ---------------------------------------------------------------------------

def show_history(session, class_date):
    """History tab: saved attendance matching the filters."""
    try:
        html, count = attendance_actions.history_html(session, clean_date(class_date))
        return html, f"{count} attendance row(s)."
    except Exception as error:
        return "", f"⚠️ {error}"


def export_history(session, class_date):
    """Export CSV button on the History tab."""
    try:
        path = attendance_store.export_csv(session, clean_date(class_date))
        return gr.File(value=path, visible=True), "✅ CSV ready below."
    except Exception as error:
        return gr.File(value=None, visible=False), f"⚠️ {error}"


# ---------------------------------------------------------------------------
# Layout: what the page looks like. Each gr.Tab is one tab on the page; the
# buttons are connected to the functions above in the Events section below.
# ---------------------------------------------------------------------------

with gr.Blocks(title="Class Attendance") as demo:
    gr.HTML(
        '<div class="att-header"><h1>📋 Class Attendance</h1>'
        "<p>Take attendance from a photo, a video or the live camera. "
        "Enrolled students are marked Present automatically.</p></div>"
    )
    with gr.Row(equal_height=True):
        attendance_session = gr.Dropdown(
            label="Class / session", value="COSC330 / Lecture 1", scale=3,
            choices=["COSC330 / Lecture 1"], allow_custom_value=True, filterable=True,
            info="Pick a class, or type a new name (e.g. COSC330 / Lecture 2) to start a new session.",
        )
        attendance_date = gr.DateTime(
            label="Class date", value=today, include_time=False, type="string", scale=2,
        )

    with gr.Tab("Attendance Sheet") as sheet_tab:
        with gr.Row():
            sheet_refresh = gr.Button("🔄 Refresh", scale=0, min_width=170)
            sheet_export = gr.Button("⬇️ Export CSV", scale=0, min_width=170)
            sheet_status = gr.Markdown()
        sheet_html = gr.HTML()
        sheet_file = gr.File(label="Attendance CSV", interactive=False, visible=False)

        sheet_unknowns = gr.HTML()

        with gr.Accordion("✏️ Edit attendance", open=False):
            with gr.Row(equal_height=True):
                edit_student = gr.Dropdown(label="Student", choices=[], scale=3)
                mark_present = gr.Button("✅ Mark Present", scale=1)
                mark_absent = gr.Button("❌ Mark Absent", scale=1)
            edit_status = gr.Markdown()

        with gr.Accordion("🗑️ Delete records", open=False):
            gr.Markdown(
                "**Clear this class** removes the attendance and unknown faces for the "
                "class and date chosen at the top. **Delete all history** removes every "
                "class. Enrolled students are never deleted. This can't be undone."
            )
            confirm_delete = gr.Checkbox(label="I understand this can't be undone")
            with gr.Row():
                clear_class_button = gr.Button("Clear this class", variant="stop", scale=0, min_width=200)
                delete_all_button = gr.Button("Delete all history", variant="stop", scale=0, min_width=200)
            delete_status = gr.Markdown()

    with gr.Tab("Take Attendance"):
        with gr.Tab("📷 Photo"):
            with gr.Row():
                with gr.Column(scale=2):
                    # gr.File passes the original upload straight to OpenCV, so an
                    # unreadable image reaches analyze_photo and gets a clear message.
                    photo_input = gr.File(
                        label="Classroom photo", file_types=["image"], type="filepath"
                    )
                    photo_button = gr.Button("Analyze Photo", variant="primary", size="lg")
                    photo_message = gr.Markdown()
                    photo_annotated = gr.Image(label="Result", interactive=False, type="pil")
                    photo_faces = gr.HTML()
                with gr.Column(scale=3):
                    photo_sheet = gr.HTML()
                    photo_unknowns = gr.HTML()

        with gr.Tab("🎞️ Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.Video(label="Classroom video")
                    video_button = gr.Button("Analyze Video", variant="primary", size="lg")
                    video_message = gr.Markdown()
                with gr.Column(scale=3):
                    video_sheet = gr.HTML()
                    video_unknowns = gr.HTML()

        with gr.Tab("🎥 Live Camera"):
            gr.Markdown(
                "Live attendance starts automatically when you open this tab (allow camera "
                f"access the first time). A frame is checked about every "
                f"{LIVE_SECONDS_BETWEEN_FRAMES} seconds and recognized students are marked "
                "Present right away. Nothing is recorded or saved except the attendance."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    live_camera = gr.Image(
                        label="Camera", sources=["webcam"], type="numpy", streaming=True,
                        elem_id="live-cam",
                    )
                    live_annotated = gr.Image(label="Last checked frame", interactive=False)
                    live_message = gr.Markdown()
                with gr.Column(scale=3):
                    live_sheet = gr.HTML()
                    live_unknowns = gr.HTML()

    with gr.Tab("History") as history_tab:
        gr.Markdown(
            "Every saved class, with Present and Absent students and the Unknown faces "
            "detected (also included in the CSV). Leave a filter blank "
            "to include everything. Times are processing times in UTC+04:00."
        )
        with gr.Row(equal_height=True):
            filter_session = gr.Textbox(label="Class / session (exact match)", scale=3)
            filter_date = gr.Textbox(label="Class date (YYYY-MM-DD)", scale=2)
        with gr.Row():
            history_refresh = gr.Button("🔄 Refresh", scale=0, min_width=170)
            history_export = gr.Button("⬇️ Export CSV", scale=0, min_width=170)
            history_status = gr.Markdown()
        history_html = gr.HTML()
        history_file = gr.File(label="Attendance CSV", interactive=False, visible=False)

    with gr.Tab("Students"):
        with gr.Tab("Register Student"):
            with gr.Row():
                with gr.Column():
                    reg_id = gr.Textbox(label="Student ID")
                    reg_name = gr.Textbox(label="Full Name")
                    reg_photo = gr.Image(
                        label="Face Photo", sources=["upload", "webcam"], type="pil"
                    )
                    reg_button = gr.Button("Register Student", variant="primary")
                with gr.Column():
                    reg_message = gr.Markdown()
                    reg_face = gr.Image(label="Saved face crop", interactive=False)

        with gr.Tab("Registered Students") as students_tab:
            with gr.Row():
                refresh_button = gr.Button("🔄 Refresh List", scale=0, min_width=160)
                rebuild_button = gr.Button("Rebuild Gallery", scale=0, min_width=160)
            gallery_message = gr.Markdown()
            students_table = gr.Dataframe(
                headers=TABLE_HEADERS, label="Registered students", interactive=False
            )

    # ---- Events: which function runs when a button is pressed ---------------
    class_inputs = [attendance_session, attendance_date]
    all_sheets = [sheet_html, photo_sheet, video_sheet, live_sheet]
    all_unknowns = [sheet_unknowns, live_unknowns]

    def show_all_sheets(session, class_date):
        html = show_sheet(session, class_date)
        faces = class_unknowns(session, class_date)
        return [html] * len(all_sheets) + [faces] * len(all_unknowns)

    def show_sheet_tab(session, class_date):
        return show_sheet(session, class_date), class_unknowns(session, class_date)

    # The sheet always follows the class chosen at the top of the page.
    demo.load(show_all_sheets, class_inputs, all_sheets + all_unknowns)
    demo.load(session_choices, attendance_session, attendance_session)
    demo.load(student_choices, None, edit_student)
    attendance_session.change(show_all_sheets, class_inputs, all_sheets + all_unknowns)
    attendance_date.change(show_all_sheets, class_inputs, all_sheets + all_unknowns)
    sheet_tab.select(show_sheet_tab, class_inputs, [sheet_html, sheet_unknowns])
    sheet_tab.select(student_choices, None, edit_student)
    sheet_refresh.click(show_sheet_tab, class_inputs, [sheet_html, sheet_unknowns])
    sheet_export.click(export_sheet, class_inputs, [sheet_file, sheet_status])

    mark_present.click(lambda sid, s, d: change_status(sid, True, s, d),
                       [edit_student] + class_inputs, [edit_status, sheet_html])
    mark_absent.click(lambda sid, s, d: change_status(sid, False, s, d),
                      [edit_student] + class_inputs, [edit_status, sheet_html])
    delete_outputs = [delete_status, sheet_html, sheet_unknowns, confirm_delete]
    clear_class_button.click(clear_this_class, [confirm_delete] + class_inputs, delete_outputs)
    delete_all_button.click(delete_everything, [confirm_delete] + class_inputs, delete_outputs)

    photo_button.click(
        analyze_photo, inputs=[photo_input] + class_inputs,
        outputs=[photo_message, photo_annotated, photo_faces, photo_sheet, photo_unknowns],
        concurrency_id="recognition", concurrency_limit=1,
    )
    video_button.click(
        analyze_video, inputs=[video_input] + class_inputs,
        outputs=[video_message, video_sheet, video_unknowns],
        concurrency_id="recognition", concurrency_limit=1,
    )
    live_camera.stream(
        live_frame, inputs=[live_camera] + class_inputs,
        outputs=[live_annotated, live_message, live_sheet, live_unknowns],
        stream_every=LIVE_SECONDS_BETWEEN_FRAMES,
        concurrency_id="recognition", concurrency_limit=1, show_progress="hidden",
    )

    history_inputs = [filter_session, filter_date]
    demo.load(show_history, history_inputs, [history_html, history_status])
    # Reload whenever the tab is opened, so new attendance shows without a refresh.
    history_tab.select(show_history, history_inputs, [history_html, history_status])
    history_refresh.click(show_history, history_inputs, [history_html, history_status])
    history_export.click(export_history, history_inputs, [history_file, history_status])

    reg_button.click(
        register_student,
        inputs=[reg_id, reg_name, reg_photo],
        outputs=[reg_message, reg_face],
    )
    refresh_button.click(load_registered_students, outputs=students_table)
    students_tab.select(load_registered_students, outputs=students_table)
    rebuild_button.click(rebuild_gallery_clicked, outputs=gallery_message)


# Start the web server. Open the address printed in Terminal
# (http://127.0.0.1:7860) in a browser.
if __name__ == "__main__":
    ensure_database()
    demo.queue(default_concurrency_limit=1).launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=attendance_actions.APP_CSS,
        js=LIVE_CAMERA_JS,
        # The Attendance folder sits beside FaceRecognition, so Gradio must be
        # allowed to serve the CSV exports and unknown-face pictures from it.
        allowed_paths=[str(attendance_store.ATTENDANCE_DB.parent)],
    )
