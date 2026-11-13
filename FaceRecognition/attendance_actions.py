"""Glue between recognition and attendance (Student 3).

Flow for a photo (process_photo) or video (process_video):
  1. recognize_classroom.py returns a summary: which faces were found and
     which student, if any, each one is.
  2. refine() applies the threshold and the "one face per student" rule.
  3. record() saves the recognized students as Present (attendance_store.py).
  4. Unknown faces are cut out and saved, so they appear in History and CSV.
  5. The HTML views below build what the app shows: the attendance sheet,
     the list of detected faces, and the unknown-face pictures.

The views here return small HTML snippets so the app can show an attendance
sheet like a real attendance system: Present in green, Absent and Unknown in red.
Their colours are defined in APP_CSS, which recognition_app.py passes to Gradio.
"""
import base64
import io
from html import escape

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import attendance_store as store

STUDENT_HEADERS = ["Student ID", "Name"]

# Page styling (colours, cards, tables) that recognition_app.py passes to Gradio.
APP_CSS = """
.att-header {padding: 18px 22px; border-radius: 14px; margin-bottom: 4px;
  background: linear-gradient(120deg, #1e3a8a, #2563eb); color: #fff;}
.att-header h1 {margin: 0; font-size: 1.6rem; color: #fff;}
.att-header p {margin: 4px 0 0; opacity: .85; color: #fff;}
.att-cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px; margin: 6px 0 14px;}
.att-card {border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: 10px 14px; background: var(--block-background-fill);}
.att-card .num {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.att-card .lbl {font-size: .8rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
.att-card.present .num {color: #16a34a;}
.att-card.absent .num, .att-card.unknown .num {color: #dc2626;}
.att-title {font-weight: 600; margin: 4px 0 8px; font-size: 1.05rem;}
.att-table-wrap {max-height: 520px; overflow: auto; border-radius: 12px;
  border: 1px solid var(--border-color-primary);}
table.att-table {width: 100%; border-collapse: collapse; font-size: .95rem;}
table.att-table th {position: sticky; top: 0; text-align: left; padding: 10px 12px;
  background: var(--block-background-fill); border-bottom: 1px solid var(--border-color-primary);}
table.att-table td {padding: 9px 12px; border-bottom: 1px solid var(--border-color-primary);}
table.att-table th, table.att-table td {border-left: none !important; border-right: none !important; border-top: none !important;}
table.att-table {border: none !important;}
table.att-table tr.present td {background: rgba(22, 163, 74, .10);}
table.att-table tr.absent td, table.att-table tr.unknown td {background: rgba(220, 38, 38, .10);}
.att-badge {display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: .8rem; font-weight: 600; color: #fff;}
.att-badge.present {background: #16a34a;}
.att-badge.absent, .att-badge.unknown {background: #dc2626;}
.att-empty {padding: 18px; text-align: center; opacity: .7;}
.att-note {font-size: .85rem; opacity: .75; margin-top: 8px;}
.att-faces {display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px;}
.att-face {width: 104px; text-align: center; font-size: .75rem; opacity: .9;}
.att-face img {width: 104px; height: 104px; object-fit: cover; border-radius: 10px;
  border: 3px solid #dc2626; display: block; margin-bottom: 3px;}
"""


# ---------------------------------------------------------------------------
# HTML views
# ---------------------------------------------------------------------------

def _cards(cards):
    """Row of summary boxes, e.g. Enrolled 5 / Present 4 / Absent 1."""
    return '<div class="att-cards">' + "".join(
        f'<div class="att-card {cls}"><div class="num">{escape(str(num))}</div>'
        f'<div class="lbl">{escape(label)}</div></div>'
        for cls, num, label in cards) + "</div>"


def _table(headers, rows, row_classes):
    """HTML table; each row gets a CSS class (present/absent/unknown) for its colour."""
    if not rows:
        return '<div class="att-table-wrap"><div class="att-empty">Nothing to show yet.</div></div>'
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(f'<tr class="{cls}">' + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                   for row, cls in zip(rows, row_classes))
    return (f'<div class="att-table-wrap"><table class="att-table"><thead><tr>{head}'
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def _badge(status):
    """Coloured pill showing Present, Absent or Unknown."""
    return f'<span class="att-badge {status.lower()}">{escape(status)}</span>'


def _time(first_recorded):
    # "2026-11-06T15:56:36+04:00" -> "15:56"
    return escape(first_recorded[11:16]) if first_recorded else "—"


def roster_html(session, class_date, unknown_count=None):
    """Attendance sheet for one class: every enrolled student, Present or Absent."""
    try:
        rows = store.class_roster(session, class_date)
    except (ValueError, FileNotFoundError) as error:
        return f'<div class="att-empty">{escape(str(error))}</div>'
    present = sum(r[4] == "Present" for r in rows)
    absent = len(rows) - present
    rate = f"{round(100 * present / len(rows))}%" if rows else "—"
    cards = [("", len(rows), "Enrolled"), ("present", present, "Present"),
             ("absent", absent, "Absent"), ("", rate, "Attendance")]
    if unknown_count is not None:
        cards.append(("unknown", unknown_count, "Unknown faces"))
    table = _table(["Student ID", "Name", "Status", "Time recorded"],
                   [[escape(r[2]), escape(r[3]), _badge(r[4]), _time(r[5])] for r in rows],
                   [r[4].lower() for r in rows])
    title = f'<div class="att-title">{escape(session.strip())} · {escape(class_date)}</div>'
    note = ""
    if unknown_count:
        note = ('<div class="att-note">Unknown faces were detected but are not enrolled, '
                "so they are not on the attendance sheet.</div>")
    return title + _cards(cards) + table + note


def unknown_faces_html(faces, title="Unknown faces"):
    """Thumbnails of unknown faces. `faces` is [(PIL image or file path, caption)]."""
    head = (f'<div class="att-title">{escape(title)} '
            f'<span class="att-badge unknown">{len(faces)}</span></div>')
    if not faces:
        return head + '<div class="att-note">No unknown faces.</div>'
    tiles = []
    for picture, caption in faces:
        try:
            img = picture if isinstance(picture, Image.Image) else Image.open(picture)
            img = img.convert("RGB")
            img.thumbnail((208, 208))
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=85)
        except OSError:
            continue
        data = base64.b64encode(buffer.getvalue()).decode()
        tiles.append(f'<div class="att-face"><img src="data:image/jpeg;base64,{data}" '
                     f'alt="Unknown face">{escape(caption)}</div>')
    return head + '<div class="att-faces">' + "".join(tiles) + "</div>"


def faces_html(summary):
    """One row per detected face: green when recognized, red when Unknown."""
    rows, classes = [], []
    for face in summary.get("faces", []):
        sid = face.get("student_id")
        status = "Present" if sid is not None else "Unknown"
        distance = face.get("distance")
        rows.append([escape(str(face.get("face_number", ""))),
                     escape(str(sid)) if sid is not None else "—",
                     escape(face.get("name", "Unknown")),
                     _badge("Present" if sid is not None else "Unknown").replace(
                         ">Present<", ">Recognized<"),
                     f"{distance:.3f}" if isinstance(distance, (int, float)) else ""])
        classes.append(status.lower())
    return ('<div class="att-title">Faces detected in this photo</div>'
            + _table(["Face", "Student ID", "Name", "Result", "Distance"], rows, classes))


def _thumb(path, size=56):
    """Small picture of an unknown face, embedded in the page as base64."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((size * 2, size * 2))
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=80)
    except OSError:
        return ""
    data = base64.b64encode(buffer.getvalue()).decode()
    return (f'<img src="data:image/jpeg;base64,{data}" alt="Unknown face" '
            f'style="width:{size}px;height:{size}px;object-fit:cover;border-radius:8px;'
            f'border:2px solid #dc2626;vertical-align:middle;margin-right:8px">')


def history_html(session="", class_date=""):
    """Saved attendance for every class meeting that matches the filters,
    including one red row per unknown face detected in that class."""
    rows = store.attendance_rows(session, class_date)
    present = sum(r[4] == "Present" for r in rows)
    absent = sum(r[4] == "Absent" for r in rows)
    unknown = sum(r[4] == "Unknown" for r in rows)
    meetings = len({(r[0], r[1]) for r in rows})
    cards = [("", meetings, "Classes"), ("present", present, "Present"),
             ("absent", absent, "Absent"), ("unknown", unknown, "Unknown faces")]
    body = []
    for r in rows:
        name = escape(r[3])
        if r[4] == "Unknown":
            name = _thumb(r[6]) + name
        body.append([escape(r[0]), escape(r[1]), escape(r[2]), name, _badge(r[4]), _time(r[5])])
    table = _table(["Class", "Date", "Student ID", "Name", "Status", "Time recorded"],
                   body, [r[4].lower() for r in rows])
    return _cards(cards) + table, len(rows)


def student_table(recognized_students):
    """One row per unique recognized student."""
    rows = [[s["student_id"], s.get("name", "")] for s in recognized_students]
    return pd.DataFrame(rows, columns=STUDENT_HEADERS)


# ---------------------------------------------------------------------------
# Photo annotation
# ---------------------------------------------------------------------------

def load_photo(photo_path):
    # OpenCV applies the photo's EXIF rotation when it reads the file, so do
    # the same here or the boxes land in the wrong place on phone photos.
    with Image.open(photo_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def unknown_crops(img, summary, pad=0.25):
    """Cut out each Unknown face (with a little margin) from a PIL image."""
    crops = []
    for face in summary.get("faces", []):
        box = face.get("bounding_box")
        if face.get("student_id") is not None or not box or len(box) != 4:
            continue
        x, y, w, h = [int(v) for v in box]
        dx, dy = int(w * pad), int(h * pad)
        crop = img.crop((max(0, x - dx), max(0, y - dy),
                         min(img.width, x + w + dx), min(img.height, y + h + dy)))
        if crop.width > 4 and crop.height > 4:
            crops.append(crop)
    return crops


def annotate_photo(photo_path, summary, img=None):
    """Draw a green box and name on each student, a red box on each Unknown face."""
    img = (img or load_photo(photo_path)).copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, img.width // 60))
    for face in summary.get("faces", []):
        box = face.get("bounding_box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = [int(v) for v in box]
        known = face.get("student_id") is not None
        color = "#16a34a" if known else "#dc2626"
        draw.rectangle((x, y, x + w, y + h), outline=color, width=max(2, img.width // 300))
        label = face.get("name", str(face["student_id"])) if known else "Unknown"
        draw.text((max(0, x), max(0, y - font.size - 4)), label, fill=color, font=font)
    return img


# ---------------------------------------------------------------------------
# Recognition -> attendance
# ---------------------------------------------------------------------------

def refine(summary, threshold):
    """Make matching stricter after recognition.

    1. A face whose distance is above `threshold` becomes Unknown.
    2. One student can only be one face in the same picture: if two faces
       match the same student, only the closer one keeps the match. (This
       stops look-alikes, such as relatives, from being marked as that student.)
    Video summaries have no per-face list, so they are returned unchanged.
    """
    faces = summary.get("faces")
    if faces is None:
        return summary
    best = {}
    for face in faces:
        sid = face.get("student_id")
        if sid is None:
            continue
        distance = face.get("distance", 0.0)
        if distance > threshold:
            face.update(student_id=None, name="Unknown")
            continue
        if sid in best:
            loser = face if best[sid]["distance"] <= distance else best[sid]
            if loser is best[sid]:
                best[sid] = face
            loser.update(student_id=None, name="Unknown")
        else:
            best[sid] = face
    summary["recognized_students"] = [
        {"student_id": f["student_id"], "name": f["name"]}
        for f in faces if f.get("student_id") is not None
    ]
    summary["unknown_count"] = sum(f.get("student_id") is None for f in faces)
    summary["total_faces"] = len(faces)
    return summary


def students_message(recognized_students):
    """Text naming the students found, or saying none were recognized."""
    count = len(recognized_students)
    if count == 0:
        return ("**No enrolled students were recognized.**\n\n"
                "Faces may have been detected, but none matched the enrollment gallery.")
    names = ", ".join(s.get("name", s["student_id"]) for s in recognized_students)
    return f"**Unique recognized students: {count}** ({names})"


def save_message(saved):
    """Text saying how many students were newly marked Present."""
    message = (f"✅ Attendance saved: {saved['added']} newly marked Present, "
               f"{saved['already_present']} already Present.")
    if saved["skipped_ids"]:
        message += ("\n\nNot recorded: " + ", ".join(saved["skipped_ids"])
                    + " (not in students.db).")
    return message


def record(summary, session, class_date, kind="photo"):
    """Save attendance for one recognition summary and build the status text.

    Detections are counted per face (and per sampled frame for video), so the
    same student can add several detections but is only one attendance row.
    """
    saved = store.mark_attendance(summary["recognized_students"], session, class_date)
    lines = [students_message(summary["recognized_students"])]
    if "frames_sampled" in summary:
        lines.append(f"Frames sampled: {summary['frames_sampled']}")
    where = {"photo": "in photo", "video": "across sampled frames",
             "live": "in this camera frame"}[kind]
    lines.append(f"Face detections {where}: {summary['total_faces']} "
                 f"(unknown: {summary['unknown_count']})")
    lines.append(save_message(saved))
    return "\n\n".join(lines)


def process_photo(path, session, class_date, recognizer, threshold=1.0):
    """Take attendance from one classroom photo (the Photo tab).

    recognizer is recognize_classroom.recognize_image. Returns the status
    text, the photo with boxes, the face list, the attendance sheet and the
    unknown-face pictures, in the order the app displays them.
    """
    # Steps 1-2: recognize the faces, then apply the threshold rules.
    summary = refine(recognizer(path, save_output=False), threshold)
    # Prepare display before saving, so an unreadable preview cannot mark attendance.
    original = load_photo(path)
    picture = annotate_photo(path, summary, original)
    crops = unknown_crops(original, summary)
    # Steps 3-4: save Present students and the unknown faces.
    message = record(summary, session, class_date, "photo")
    store.save_unknown_faces(crops, session, class_date, "Photo")
    return (message, picture, faces_html(summary),
            roster_html(session, class_date, summary["unknown_count"]),
            unknown_faces_html([(c, f"Face {i}") for i, c in enumerate(crops, 1)],
                               "Unknown faces in this photo"))


def process_video(path, session, class_date, recognizer):
    """Take attendance from a classroom video (the Video tab).

    recognizer is recognize_classroom.recognize_video. Returns the status
    text, the attendance sheet and one picture per unknown person.
    """
    # recognize_video's defaults: one frame every 2 seconds through the whole
    # video (early stopping is off unless stop_when_all_found=True is passed).
    summary = recognizer(path)
    message = record(summary, session, class_date, "video")
    if summary.get("stopped_early"):
        message += "\n\nStopped early: every enrolled student was already found."
    # recognize_video groups unknown detections across frames, so this is one
    # picture per unknown person (not one per frame).
    people = summary.get("unknown_faces") or []
    store.save_unknown_faces(people, session, class_date, "Video")
    message += f"\n\nUnknown people in this video: {len(people)}"
    unknown = unknown_faces_html([(c, f"Person {i}") for i, c in enumerate(people, 1)],
                                 "Unknown people in this video")
    return message, roster_html(session, class_date, len(people)), unknown
