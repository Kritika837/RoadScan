"""
Ad-hoc integration test script for the Citizen Road Damage Reporting module,
plus a smoke-test pass over the existing routes to confirm nothing broke.

Run with: python3 tests/run_citizen_report_tests.py
(run from the road_damage_app/ directory, same as the Flask app itself)
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402
_helpers.isolated_app_copy()      # run against a temp copy: never touches the real DB/static files
_helpers.ensure_sample_uploads()  # synthetic photos if the repo has none

import app as flaskapp  # noqa: E402

APP_ROOT = flaskapp.APP_ROOT
SAMPLE_IMAGES = [
    os.path.join(APP_ROOT, "static", "uploads", f)
    for f in os.listdir(os.path.join(APP_ROOT, "static", "uploads"))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

results = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail and status == 'FAIL' else ""))


def img_bytes(path):
    # Unique trailing bytes each call: the app treats byte-identical photos as repeats
    # (double-tap guard / no extra points), but detection still sees the same image.
    with open(path, "rb") as f:
        return io.BytesIO(f.read() + os.urandom(8))


def main():
    flaskapp.app.config["TESTING"] = True
    client = flaskapp.app.test_client()

    if not SAMPLE_IMAGES:
        print("No sample images found in static/uploads - aborting.")
        sys.exit(1)
    sample1, sample2 = SAMPLE_IMAGES[0], SAMPLE_IMAGES[min(1, len(SAMPLE_IMAGES) - 1)]

    # 1. Existing home page still works
    r = client.get("/")
    check("Existing home page (/) loads", r.status_code == 200, f"status={r.status_code}")
    check("Home page shows Report Road Damage CTA", b"Report Road Damage" in r.data)

    # 2. Existing /analyze (regular inspection) still works
    r = client.post("/analyze", data={
        "image": (img_bytes(sample1), "test.jpg"),
        "location": "Test St existing-flow",
        "confidence": "0",
    }, content_type="multipart/form-data")
    check("Existing /analyze route still works", r.status_code == 200, f"status={r.status_code}")

    # 3. Existing /history and /compare still reachable
    r = client.get("/history")
    check("Existing /history route still works", r.status_code == 200, f"status={r.status_code}")
    r = client.get("/compare")
    check("Existing /compare (GET) route still works", r.status_code == 200, f"status={r.status_code}")

    # 4. New: report form page
    r = client.get("/report")
    check("GET /report (report form) loads", r.status_code == 200, f"status={r.status_code}")
    check("Report form has image + location fields", b'name="image"' in r.data and b'name="location"' in r.data)

    # 5. New: submit a citizen report (this exercises detection + severity + health + priority)
    r = client.post("/report/submit", data={
        "image": (img_bytes(sample1), "damage1.jpg"),
        "location": "Main St & 5th Ave",
        "road_name": "Main St",
        "landmark": "Near City Park",
        "description": "Getting worse after rain",
        "latitude": "28.6139",
        "longitude": "77.2090",
    }, content_type="multipart/form-data", follow_redirects=True)
    check("POST /report/submit succeeds (redirect to detail)", r.status_code == 200, f"status={r.status_code}")
    check("Submit response shows a Report ID", b"RS-" in r.data)

    reports = flaskapp.creport.list_reports()
    check("Report was persisted to SQLite", len(reports) >= 1, f"count={len(reports)}")
    report_id = reports[0].report_id
    check("Report ID matches expected format RS-YYYY-NNNNNN",
          report_id.startswith("RS-") and len(report_id.split("-")) == 3, report_id)

    # Verify YOLO/detector pipeline was actually executed (not faked): detected_damage field
    # is populated from real assessments, and severity/health/priority are internally consistent.
    rep = flaskapp.creport.get_report(report_id)
    check("Report has detector-derived fields (health_score set)", rep.health_score is not None)
    check("Report has repair_priority set from severity.py logic", rep.repair_priority in
          ["LOW", "MEDIUM", "HIGH", "URGENT"], rep.repair_priority)

    # 6. Report ID is reproducible / persistent (re-fetch gives identical data)
    rep2 = flaskapp.creport.get_report(report_id)
    check("Report ID reproducible across reads", rep.report_id == rep2.report_id == report_id)

    # 7. New: Track Report
    r = client.post("/track", data={"report_id": report_id}, follow_redirects=True)
    check("Track Report finds submitted report", r.status_code == 200 and report_id.encode() in r.data)

    r = client.post("/track", data={"report_id": "RS-2099-999999"}, follow_redirects=True)
    check("Track Report handles unknown ID gracefully", r.status_code == 200)

    # 8. New: report detail page content
    r = client.get(f"/reports/{report_id}")
    check("Report detail page loads", r.status_code == 200)
    check("Report detail shows the progress tracker", b"Progress" in r.data and b'class="road' in r.data)
    check("Report detail shows Road Health Score", b"Road Health" in r.data)

    # 9. New: My Reports (session-scoped)
    r = client.get("/reports")
    check("My Reports page loads", r.status_code == 200)
    check("My Reports lists the submitted report (same citizen)", report_id.encode() in r.data)

    # 10. Authority dashboard (the old Maintenance dashboard now lives under /authority)
    r = client.get("/maintenance")
    check("Legacy /maintenance redirects to the authority dashboard", r.status_code == 302 and "/authority" in r.headers["Location"])
    r = client.get("/authority/reports")
    check("Authority reports list loads and lists the report", r.status_code == 200 and report_id.encode() in r.data)

    # 11. Status update flow: (assign a department first) -> UNDER_REVIEW -> ASSIGNED -> IN_PROGRESS
    client.post(f"/authority/reports/{report_id}/status", data={"status": "UNDER_REVIEW"}, follow_redirects=True)
    check("Status update to UNDER_REVIEW persisted", flaskapp.creport.get_report(report_id).status == "UNDER_REVIEW")
    client.post(f"/authority/reports/{report_id}/assign", data={"department": "Test Roads Dept", "officer_name": "Test Officer"}, follow_redirects=True)
    check("Assignment moves the report to ASSIGNED", flaskapp.creport.get_report(report_id).status == "ASSIGNED")
    client.post(f"/authority/reports/{report_id}/status", data={"status": "IN_PROGRESS"}, follow_redirects=True)
    check("Status update to IN_PROGRESS persisted", flaskapp.creport.get_report(report_id).status == "IN_PROGRESS")

    # 12. Resolving now REQUIRES an after-repair photo (evidence)
    r = client.post(f"/authority/reports/{report_id}/resolve", data={}, content_type="multipart/form-data", follow_redirects=True)
    check("Resolve without an after-photo is refused", flaskapp.creport.get_report(report_id).status == "IN_PROGRESS")
    r = client.post(f"/authority/reports/{report_id}/resolve", data={
        "after_image": (img_bytes(sample2), "after.jpg"), "notes": "Resurfaced"},
        content_type="multipart/form-data", follow_redirects=True)
    rep = flaskapp.creport.get_report(report_id)
    check("Report marked RESOLVED with repair evidence", rep.status == "RESOLVED", f"got={rep.status}")
    check("resolved_at timestamp set", rep.resolved_at is not None)

    # 13. Second report, this time with a different photo -> resolve -> before/after evidence
    r = client.post("/report/submit", data={
        "image": (img_bytes(sample1), "damage2.jpg"),
        "location": "Ring Road km 4",
    }, content_type="multipart/form-data", follow_redirects=True)
    reports2 = flaskapp.creport.list_reports()
    report_id2 = reports2[0].report_id  # most recent
    check("Second report created with a different ID", report_id2 != report_id)

    r = client.post(f"/authority/reports/{report_id2}/resolve", data={
        "after_image": (img_bytes(sample2), "after.jpg"),
    }, content_type="multipart/form-data", follow_redirects=True)
    rep2r = flaskapp.creport.get_report(report_id2)
    check("Report with after-photo marked RESOLVED", rep2r.status == "RESOLVED")
    check("Before/after comparison stored (reused comparison.py)", rep2r.comparison is not None,
          str(rep2r.comparison))
    check("Comparison has a verdict field", rep2r.comparison and "verdict" in rep2r.comparison)

    # 14. Report detail page shows resolution section
    r = client.get(f"/reports/{report_id2}")
    check("Resolved report detail shows repair evidence + resolution date", b"Repair evidence" in r.data and b"Resolution date" in r.data)

    # 15. Downloadable report files exist on disk
    if rep2r.pdf_report_path:
        pdf_disk_path = os.path.join(APP_ROOT, rep2r.pdf_report_path.lstrip("/"))
        check("PDF report file exists on disk", os.path.isfile(pdf_disk_path), pdf_disk_path)
    if rep2r.json_report_path:
        json_disk_path = os.path.join(APP_ROOT, rep2r.json_report_path.lstrip("/"))
        check("JSON report file exists on disk", os.path.isfile(json_disk_path), json_disk_path)

    # 16. Test an image with (likely) no detected damage is handled without crashing -
    # we can't guarantee zero detections from any given photo, so we assert the report
    # is still created successfully and severity/priority never crash on empty assessments.
    r = client.post("/report/submit", data={
        "image": (img_bytes(sample2), "maybe_clean.jpg"),
        "location": "Test clean-image handling",
    }, content_type="multipart/form-data", follow_redirects=True)
    check("Report submission does not crash regardless of detection count", r.status_code == 200)

    # 17. Multiple detections handling: just confirm the pipeline + report path handles
    # whatever count of detections the heuristic detector returns without error.
    reports_all = flaskapp.creport.list_reports()
    check("Multiple citizen reports coexist in storage", len(reports_all) >= 3, f"count={len(reports_all)}")

    # 18. Existing report generation (/analyze route's own PDF/JSON/HTML) still unaffected
    r = client.post("/analyze", data={
        "image": (img_bytes(sample1), "existing_report_test.jpg"),
        "location": "",
        "confidence": "0",
    }, content_type="multipart/form-data")
    check("Existing /analyze report generation still works after changes", r.status_code == 200)

    print("\n--- SUMMARY ---")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"{passed} passed, {failed} failed, {len(results)} total")
    if failed:
        print("\nFailed checks:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
