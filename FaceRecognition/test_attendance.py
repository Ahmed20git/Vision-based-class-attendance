"""Automated tests for the attendance code (attendance_store / attendance_actions).

Each test uses a temporary students.db and attendance.db, so your real
records are never touched. Face recognition is not run here; the tests feed in
ready-made recognition results.

Run from the FaceRecognition folder:
    ./.venv/bin/python -m unittest test_attendance -v
"""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import attendance_store as store
import attendance_actions as actions

class AttendanceTests(unittest.TestCase):
    """One test per behaviour the attendance system must have."""
    def setUp(self):
        """Create a temporary database with two enrolled students, 001 and 002."""
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.students=self.root/"students.db"
        self.db=self.root/"attendance.db"
        with sqlite3.connect(self.students) as c:
            c.execute("CREATE TABLE students(student_id TEXT PRIMARY KEY,student_name TEXT,face_image TEXT)")
            c.executemany("INSERT INTO students VALUES(?,?,?)",[("001","Student A","a.jpg"),("002","Student B","b.jpg")])
        self.patches=[patch.object(store,"ATTENDANCE_DB",self.db),patch.object(store,"STUDENTS_DB",self.students)]
        for p in self.patches:p.start()
    def tearDown(self):
        """Delete the temporary databases."""
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def test_duplicate_session_date_and_first_time(self):
        """A student is saved once per class, and keeps their first time."""
        people=[{"student_id":"001"},{"student_id":"001"},{"student_id":None}]
        self.assertEqual(store.mark_attendance(people,"COSC / L1","2026-11-04")["added"],1)
        first=store.attendance_rows()[0][-1]
        self.assertEqual(store.mark_attendance(people,"COSC / L1","2026-11-04")["added"],0)
        self.assertEqual(store.attendance_rows()[0][-1],first)
        store.mark_attendance(people,"COSC / L2","2026-11-04")
        store.mark_attendance(people,"COSC / L1","2026-11-05")
        self.assertEqual(len(store.attendance_rows()),6)  # 3 classes x 2 enrolled
        rows=store.attendance_rows("COSC / L1","2026-11-04")
        self.assertEqual([(r[2],r[4]) for r in rows],[("001","Present"),("002","Absent")])
    def test_unknown_stale_gallery_and_database_name(self):
        """Unknown faces and IDs not in students.db are skipped; names come from students.db."""
        r=store.mark_attendance([{"student_id":None},{"student_id":"missing"},{"student_id":"001","name":"Wrong name"}],"L1","2026-11-04")
        self.assertEqual(r["skipped_ids"],["missing"])
        self.assertEqual(store.attendance_rows()[0][2:4],["001","Student A"])
    def test_validation_prevents_partial_write(self):
        """A bad session or date saves nothing at all."""
        for session,day in [("","2026-11-04"),("L1","bad date")]:
            with self.assertRaises(ValueError):
                store.mark_attendance([{"student_id":"001"}],session,day)
        with self.assertRaises(ValueError):
            store.mark_attendance([{"student_id":"001"},"bad"],"L1","2026-11-04")
        self.assertEqual(store.attendance_rows(),[])
    def test_csv_persistence_and_formula_escaping(self):
        """CSV export has the right rows and cannot run Excel formulas."""
        store.mark_attendance([{"student_id":"001"}],"=formula","2026-11-04")
        path=store.export_csv("=formula","2026-11-04",output_dir=self.root)
        with open(path,encoding="utf-8-sig",newline="") as f: rows=list(csv.reader(f))
        self.assertEqual(rows[1][0],"'=formula")
        self.assertEqual(rows[1][2],"001")
        self.assertEqual(len(store.attendance_rows(db=self.db)),2)
        self.assertEqual(rows[2][4],"Absent")
    def test_photo_summary_to_annotated_image_and_attendance(self):
        """A photo result draws boxes and marks the right student Present."""
        path=self.root/"photo.jpg";Image.new("RGB",(100,100),"white").save(path)
        summary={"total_faces":2,"unknown_count":1,"recognized_students":[{"student_id":"001","name":"Student A"}],
                 "faces":[{"face_number":1,"student_id":"001","name":"Student A","distance":0.1,"bounding_box":[5,5,30,30]},
                          {"face_number":2,"student_id":None,"name":"Unknown","distance":0.7,"bounding_box":[50,5,30,30]}]}
        message,image,faces,sheet,unknown=actions.process_photo(str(path),"L1","2026-11-04",lambda *a,**k:summary)
        self.assertEqual(unknown.count("data:image/jpeg"),1);self.assertEqual(len(store.unknown_faces("L1","2026-11-04",db=self.db)),1)
        self.assertEqual(image.size,(100,100))
        self.assertIn("Unknown",faces);self.assertIn('class="unknown"',faces)
        self.assertIn('<tr class="present"><td>001',sheet);self.assertIn('<tr class="absent"><td>002',sheet)
        self.assertIn("Unknown faces",sheet)
        self.assertIn("1 newly marked Present",message);self.assertIn("Unique recognized students: 1",message)
    def test_roster_lists_absent_students_and_escapes_names(self):
        """The sheet lists absent students too, and names are shown safely."""
        with sqlite3.connect(self.students) as c:
            c.execute("INSERT INTO students VALUES('10','<b>Bold</b>','c.jpg')")
        sheet=actions.roster_html("L1","2026-11-04")
        self.assertIn("&lt;b&gt;Bold&lt;/b&gt;",sheet);self.assertEqual(sheet.count('<tr class="absent">'),3)
        rows=store.class_roster("L1","2026-11-04")
        self.assertEqual([r[2] for r in rows],["001","002","10"])
    def test_photo_exif_rotation_matches_opencv(self):
        """Rotated phone photos are turned the same way OpenCV turns them."""
        path=self.root/"rotated.jpg"
        img=Image.new("RGB",(200,100),"white");exif=img.getexif();exif[0x0112]=6  # rotate 90
        img.save(path,exif=exif)
        self.assertEqual(actions.annotate_photo(path,{"faces":[]}).size,(100,200))
    def test_no_students_message(self):
        """A clear message when no enrolled student is recognized."""
        summary={"total_faces":1,"unknown_count":1,"recognized_students":[],"faces":[]}
        message=actions.record(summary,"L1","2026-11-04","live")
        self.assertIn("No enrolled students were recognized",message)
        self.assertEqual(store.attendance_rows(),[])
    def test_video_summary_counts_unique_students(self):
        """A video counts each student once, however many frames they appear in."""
        summary={"total_faces":3,"unknown_count":1,"frames_sampled":2,
                 "recognized_students":[{"student_id":"001","name":"A"},{"student_id":"001","name":"A"}]}
        summary["unknown_faces"]=[Image.new("RGB",(30,30)),Image.new("RGB",(30,30))];summary["stopped_early"]=True
        message,sheet,unknown=actions.process_video("video.mp4","L1","2026-11-04",lambda p:summary)
        self.assertIn("Unknown people in this video: 2",message);self.assertIn("Stopped early",message)
        self.assertEqual(unknown.count("data:image/jpeg"),2)
        self.assertEqual(len(store.unknown_faces("L1","2026-11-04",db=self.db)),2)
        self.assertIn("Frames sampled: 2",message);self.assertIn("across sampled frames: 3",message)
        self.assertEqual(sheet.count('<tr class="present">'),1)
        self.assertEqual(len([r for r in store.attendance_rows() if r[4]=="Present"]),1)

    def test_refine_threshold_and_one_face_per_student(self):
        """Faces above the threshold become Unknown; one student matches one face per picture."""
        summary={"faces":[
            {"face_number":1,"student_id":"001","name":"A","distance":0.36},  # look-alike
            {"face_number":2,"student_id":"001","name":"A","distance":0.18},  # real student
            {"face_number":3,"student_id":"002","name":"B","distance":0.45},  # too far
        ],"recognized_students":[],"total_faces":3,"unknown_count":0}
        r=actions.refine(summary,0.40)
        self.assertEqual([f["student_id"] for f in r["faces"]],[None,"001",None])
        self.assertEqual(r["recognized_students"],[{"student_id":"001","name":"A"}])
        self.assertEqual(r["unknown_count"],2)
        video={"recognized_students":[{"student_id":"001","name":"A"}],"total_faces":1,"unknown_count":0}
        self.assertIs(actions.refine(video,0.3),video)
    def test_manual_edit_clear_and_delete(self):
        """Mark Present / Absent, Clear this class and Delete all work."""
        store.set_status("002",True,"L1","2026-11-04")
        store.mark_attendance([{"student_id":"001"}],"L2","2026-11-04")
        self.assertEqual([r[4] for r in store.class_roster("L1","2026-11-04")],["Absent","Present"])
        store.set_status("002",False,"L1","2026-11-04")
        self.assertEqual([r[4] for r in store.class_roster("L1","2026-11-04")],["Absent","Absent"])
        with self.assertRaises(ValueError): store.set_status("999",True,"L1","2026-11-04")
        store.save_unknown_faces([Image.new("RGB",(20,20))],"L2","2026-11-04","Live",db=self.db)
        self.assertEqual(store.sessions(),["L2"])
        self.assertEqual(store.clear_class("L2","2026-11-04"),1)
        self.assertEqual(store.unknown_faces("L2","2026-11-04",db=self.db),[])
        store.mark_attendance([{"student_id":"001"}],"L3","2026-11-05")
        self.assertEqual(store.delete_all(),1);self.assertEqual(store.attendance_rows(),[])

    def test_unknown_faces_in_history_and_csv(self):
        """Unknown faces are saved and appear in History and the CSV."""
        store.mark_attendance([{"student_id":"001"}],"L1","2026-11-04")
        store.save_unknown_faces([Image.new("RGB",(20,20)),Image.new("RGB",(20,20))],"L1","2026-11-04","Video",db=self.db)
        store.save_unknown_faces([Image.new("RGB",(20,20))],"L9","2026-11-05","Photo",db=self.db)  # unknowns only
        rows=store.attendance_rows()
        self.assertEqual([r[4] for r in rows if r[0]=="L1"],["Present","Absent","Unknown","Unknown"])
        self.assertEqual([r[2] for r in rows if r[4]=="Unknown" and r[0]=="L1"],["Unknown 1","Unknown 2"])
        self.assertEqual([r[4] for r in rows if r[0]=="L9"],["Absent","Absent","Unknown"])
        self.assertTrue(all(Path(r[6]).is_file() for r in rows if r[4]=="Unknown"))
        path=store.export_csv("L1","2026-11-04",output_dir=self.root)
        with open(path,encoding="utf-8-sig",newline="") as f: csv_rows=list(csv.reader(f))
        self.assertEqual(csv_rows[0][-1],"Face image");self.assertEqual(csv_rows[3][3:5],["Unknown (Video)","Unknown"])
        html,count=actions.history_html()
        self.assertEqual(count,len(rows));self.assertEqual(html.count('<tr class="unknown">'),3)

if __name__=="__main__":
    unittest.main()
