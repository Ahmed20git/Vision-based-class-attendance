"""Attendance records: saving, reading, editing and exporting (Student 3).

This file only stores results. It does not know anything about faces or the
recognition model; recognition_app.py / attendance_actions.py pass it the
students that recognize_classroom.py found.

Data is kept in Attendance/attendance.db (SQLite) with two tables:
  attendance    - one row per (session, class date, student) who was Present.
                  Absent is not stored: it is everyone in Enrollment/students.db
                  who has no Present row for that class.
  unknown_faces - one row per saved picture of a face that matched nobody.
                  The pictures themselves are in Attendance/unknown_faces/.

CSV exports are written to Attendance/exports/.
"""
import csv
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
ATTENDANCE_DB = ROOT.parent / "Attendance" / "attendance.db"
STUDENTS_DB = ROOT.parent / "Enrollment" / "students.db"
HEADERS = ["Session", "Class date", "Student ID", "Name", "Status", "First recorded", "Face image"]

def connect(db=None):
    """Open attendance.db, creating the file and both tables the first time."""
    db = Path(db or ATTENDANCE_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS attendance (
        session TEXT NOT NULL, class_date TEXT NOT NULL, student_id TEXT NOT NULL,
        student_name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Present',
        first_recorded TEXT NOT NULL,
        PRIMARY KEY(session,class_date,student_id))""")
    con.execute("""CREATE TABLE IF NOT EXISTS unknown_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT NOT NULL, class_date TEXT NOT NULL,
        image_path TEXT NOT NULL, source TEXT NOT NULL, detected_at TEXT NOT NULL)""")
    con.commit()
    return con

def validate_session(session, class_date):
    """Check the session name and YYYY-MM-DD date before anything is saved."""
    session = (session or "").strip()
    if not session or len(session) > 120:
        raise ValueError("Enter a course / session name of 1–120 characters.")
    try:
        date.fromisoformat(class_date)
    except (ValueError, TypeError):
        raise ValueError("Enter the class date as YYYY-MM-DD.") from None
    return session, class_date

def registered_students(db=None):
    """{student_id: name} of every enrolled student, read from students.db (read-only)."""
    path = Path(db or STUDENTS_DB).resolve()
    if not path.is_file():
        raise FileNotFoundError("Enrollment/students.db is missing.")
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)) as con:
        return {str(sid): name or "" for sid, name in
                con.execute("SELECT student_id, student_name FROM students")}

def mark_attendance(recognized_students, session, class_date,
                    db=None, students_db=None):
    """Mark the recognized students Present for one class meeting.

    A student already Present keeps their first time (no duplicate rows,
    thanks to the PRIMARY KEY and INSERT OR IGNORE). IDs that are not in
    students.db are skipped. Returns how many were added, already present
    and skipped.
    """
    session, class_date = validate_session(session, class_date)
    enrolled = registered_students(students_db)
    accepted, skipped = set(), set()
    # Validate the full batch before writing.
    for student in recognized_students:
        if not isinstance(student, dict):
            raise ValueError("Recognition returned an invalid student record.")
        sid = student.get("student_id")
        if sid is None:
            continue
        sid = str(sid)
        if sid not in enrolled:
            skipped.add(sid)
        else:
            accepted.add(sid)
    # UAE time (UTC+4) for the "First recorded" column.
    timestamp = datetime.now(timezone(timedelta(hours=4))).isoformat(timespec="seconds")
    with closing(connect(db)) as con, con:
        before = con.total_changes
        for sid in sorted(accepted):
            con.execute("""INSERT OR IGNORE INTO attendance
                (session,class_date,student_id,student_name,first_recorded)
                VALUES(?,?,?,?,?)""", (session,class_date,sid,enrolled[sid],timestamp))
        added = con.total_changes-before
    return {"added": added, "already_present": len(accepted)-added,
            "skipped_ids": sorted(skipped)}

def _id_order(sid):
    """Sort student IDs numerically when they are numbers (2 before 10)."""
    return (0, int(sid), sid) if sid.isdigit() else (1, 0, sid)


def class_roster(session, class_date, db=None, students_db=None):
    """Every enrolled student for one class meeting, marked Present or Absent.

    Absent is worked out from students.db, not stored, so a student enrolled
    later still shows as Absent for earlier classes.
    """
    session, class_date = validate_session(session, class_date)
    enrolled = registered_students(students_db)
    with closing(connect(db)) as con:
        present = {r["student_id"]: r for r in con.execute(
            "SELECT student_id,student_name,first_recorded FROM attendance "
            "WHERE session=? AND class_date=?", (session, class_date))}
    rows = []
    for sid in sorted(set(enrolled) | set(present), key=_id_order):
        if sid in present:
            name = enrolled.get(sid) or present[sid]["student_name"]
            rows.append([session, class_date, sid, name, "Present",
                         present[sid]["first_recorded"]])
        else:
            rows.append([session, class_date, sid, enrolled[sid], "Absent", ""])
    return rows


def attendance_rows(session="", class_date="", db=None, students_db=None):
    """Full record for every class meeting that matches the filters.

    Each meeting lists every enrolled student (Present or Absent), then one
    row per saved unknown face (Status "Unknown", with its picture's path).
    Blank filters include everything. A meeting is listed once it has a
    Present record or an unknown face, or when both filters name it exactly.
    """
    session = (session or "").strip()
    class_date = (class_date or "").strip()
    if class_date:
        try:
            date.fromisoformat(class_date)
        except ValueError:
            raise ValueError("Filter date must be YYYY-MM-DD, or blank for all dates.") from None
    conditions, args = [], []
    if session:
        conditions.append("session=?"); args.append(session)
    if class_date:
        conditions.append("class_date=?"); args.append(class_date)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    query = ("SELECT session,class_date FROM attendance" + where + " UNION "
             "SELECT session,class_date FROM unknown_faces" + where +
             " ORDER BY class_date DESC,session")
    args = args * 2
    with closing(connect(db)) as con:
        meetings = [tuple(r) for r in con.execute(query, args)]
    if session and class_date and not meetings:
        meetings = [(session, class_date)]
    rows = []
    for meeting_session, meeting_date in meetings:
        rows += [r + [""] for r in class_roster(meeting_session, meeting_date, db, students_db)]
        rows += unknown_rows(meeting_session, meeting_date, db)
    return rows


def unknown_rows(session, class_date, db=None):
    """One attendance-style row per saved unknown face, oldest first."""
    folder = unknown_dir(db)
    with closing(connect(db)) as con:
        found = con.execute("SELECT image_path,source,detected_at FROM unknown_faces "
                            "WHERE session=? AND class_date=? ORDER BY id",
                            (session, class_date)).fetchall()
    return [[session, class_date, f"Unknown {n}", f"Unknown ({r['source']})", "Unknown",
             r["detected_at"], str(folder / r["image_path"])]
            for n, r in enumerate(found, 1)]


def export_csv(session="", class_date="", db=None, output_dir=None, students_db=None):
    """Write the matching records to a new CSV file in Attendance/exports and return its path."""
    rows = attendance_rows(session,class_date,db,students_db)
    if not rows:
        raise ValueError("No attendance records match these filters yet.")
    folder = Path(output_dir) if output_dir else ROOT.parent/"Attendance"/"exports"
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/("attendance_"+uuid4().hex+".csv")
    # Stop Excel from running a cell as a formula (CSV injection).
    def safe(value):
        value=str(value)
        return "'"+value if value.lstrip().startswith(("=","+","-","@")) else value
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f);writer.writerow(HEADERS)
        writer.writerows([[safe(v) for v in r] for r in rows])
    return str(path)


# ---------------------------------------------------------------------------
# Unknown faces: crops of faces that matched nobody, kept per class meeting
# ---------------------------------------------------------------------------

def now_text():
    """Current UAE time (UTC+4) as text, e.g. 2026-11-07T09:30:00+04:00."""
    return datetime.now(timezone(timedelta(hours=4))).isoformat(timespec="seconds")


def unknown_dir(db=None):
    """Folder where unknown-face pictures are saved."""
    return Path(db or ATTENDANCE_DB).parent / "unknown_faces"


def save_unknown_faces(crops, session, class_date, source, db=None):
    """Save PIL face crops for one class meeting. Returns how many were saved."""
    session, class_date = validate_session(session, class_date)
    if not crops:
        return 0
    folder = unknown_dir(db)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now_text()
    with closing(connect(db)) as con, con:
        for crop in crops:
            path = folder / (uuid4().hex + ".jpg")
            crop.convert("RGB").save(path, "JPEG", quality=90)
            con.execute("INSERT INTO unknown_faces(session,class_date,image_path,source,detected_at)"
                        " VALUES(?,?,?,?,?)", (session, class_date, path.name, source, stamp))
    return len(crops)


def unknown_faces(session, class_date, db=None):
    """[(image path, caption)] newest first, for one class meeting."""
    session, class_date = validate_session(session, class_date)
    folder = unknown_dir(db)
    with closing(connect(db)) as con:
        rows = con.execute("SELECT image_path,source,detected_at FROM unknown_faces "
                           "WHERE session=? AND class_date=? ORDER BY id DESC",
                           (session, class_date)).fetchall()
    return [(str(folder / r["image_path"]), f"{r['source']} · {r['detected_at'][11:16]}")
            for r in rows if (folder / r["image_path"]).is_file()]


# ---------------------------------------------------------------------------
# Editing and deleting attendance
# ---------------------------------------------------------------------------

def sessions(db=None):
    """Session names that already have records, most recent first."""
    with closing(connect(db)) as con:
        return [r[0] for r in con.execute(
            "SELECT session FROM attendance GROUP BY session ORDER BY MAX(first_recorded) DESC")]


def set_status(student_id, present, session, class_date, db=None, students_db=None):
    """Manually mark one enrolled student Present (present=True) or Absent."""
    session, class_date = validate_session(session, class_date)
    sid = str(student_id)
    with closing(connect(db)) as con, con:
        if present:
            enrolled = registered_students(students_db)
            if sid not in enrolled:
                raise ValueError(f"Student {sid} is not in students.db.")
            con.execute("""INSERT OR IGNORE INTO attendance
                (session,class_date,student_id,student_name,first_recorded)
                VALUES(?,?,?,?,?)""", (session, class_date, sid, enrolled[sid], now_text()))
        else:
            con.execute("DELETE FROM attendance WHERE session=? AND class_date=? AND student_id=?",
                        (session, class_date, sid))


def _delete_unknown(con, where, args, db):
    """Delete matching unknown-face rows and their picture files."""
    folder = unknown_dir(db)
    for (name,) in con.execute("SELECT image_path FROM unknown_faces" + where, args).fetchall():
        (folder / name).unlink(missing_ok=True)
    con.execute("DELETE FROM unknown_faces" + where, args)


def clear_class(session, class_date, db=None):
    """Delete every record and unknown face for one class meeting."""
    session, class_date = validate_session(session, class_date)
    where, args = " WHERE session=? AND class_date=?", (session, class_date)
    with closing(connect(db)) as con, con:
        removed = con.execute("DELETE FROM attendance" + where, args).rowcount
        _delete_unknown(con, where, args, db)
    return removed


def delete_all(db=None):
    """Delete all attendance history and unknown faces (students are kept)."""
    with closing(connect(db)) as con, con:
        removed = con.execute("DELETE FROM attendance").rowcount
        _delete_unknown(con, "", (), db)
    return removed
