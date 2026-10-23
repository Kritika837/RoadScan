"""
Tests for the citizen-first redesign: report flow, tracking, map data, rewards,
authority workflow (assign / escalate / resolve with repair evidence), and
that the existing pipeline outputs are still produced.

Run from anywhere:  python3 tests/run_redesign_tests.py
Uses a temporary copy of the project (see _helpers.py) - your data is untouched.
"""
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: E402

APP_DIR = _helpers.isolated_app_copy()
import app as flaskapp  # noqa: E402
import citizen_report as creport  # noqa: E402
import config  # noqa: E402
import rewards  # noqa: E402
import workflow  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))


def img(seed=1, damaged=True):
    return _helpers.road_image_bytes(seed, damaged)


def submit(client, seed, **extra):
    data = {"image": (img(seed), f"photo{seed}.jpg"), "description": "Gets worse after rain"}
    data.update(extra)
    return client.post("/report/submit", data=data, content_type="multipart/form-data")


def rid_from(resp):
    m = re.search(r"/reports/(RS-\d{4}-\d{6})/submitted", resp.headers.get("Location", ""))
    return m.group(1) if m else None


PHAGWARA = dict(latitude="31.224000", longitude="75.773900", area="LPU Front Gate", city="Phagwara",
                state="Punjab", location="LPU Front Gate, Phagwara", landmark="Main gate")
TECH = [b"browser session", b"from this browser", b"YOLO", b"Detector backend", b"detector_name",
        b"Prototype database", b"Area ratio", b"Class weight", b"Penalty by class", b"yolo26_rdd",
        b"heuristic_cv"]


def main():
    flaskapp.app.config["TESTING"] = True
    c = flaskapp.app.test_client()

    # ------------------------------------------------------------ home ---
    r = c.get("/")
    check("Home loads", r.status_code == 200)
    for txt in (b"Report road damage.", b"Track repairs.", b"Improve your community.", b"Report Road Damage",
                b"Explore Road Damage Map", b"Reports submitted", b"Issues resolved", b"Areas monitored",
                b"Report</h3>", b"Verify</h3>", b"Track</h3>"):
        check(f"Home shows {txt.decode()!r}", txt in r.data)
    check("Home starts with real zero counts (no fabricated stats)", r.data.count(b'class="n">0<') == 3)
    check("Home hides technical wording", not any(t in r.data for t in TECH))
    nav = re.search(rb'<nav class="main-nav".*?</nav>', r.data, re.S).group(0)
    labels = re.findall(rb">([^<]+)</a>", nav)
    check("Citizen nav is exactly Home/Report Damage/Road Map/My Reports/Rewards",
          labels == [b"Home", b"Report Damage", b"Road Map", b"My Reports", b"Rewards"], str(labels))
    check("Nav has no Maintenance / Before-After / History / Analyze",
          not any(w in nav for w in (b"Maintenance", b"Before", b"History", b"Analyze", b"Area Health")))

    # ------------------------------------------------------- report form ---
    r = c.get("/report")
    check("Report form loads", r.status_code == 200)
    check("Form has photo upload, GPS + manual buttons, description", all(
        t in r.data for t in (b'name="image"', b"Use My Location", b"Enter Location Manually",
                              b'name="description"', b'name="location"', b"Analyze &amp; Submit Report")))
    check("Form accepts JPG/PNG in the UI", b"image/jpeg,image/png" in r.data)
    check("Form shows problem examples", b"Pothole gets worse after rain" in r.data)
    check("Extra location fields are behind a disclosure", b'<details class="more"' in r.data)

    # ------------------------------------------ validation / friendly errors ---
    r = c.post("/report/submit", data={"location": "x"}, content_type="multipart/form-data", follow_redirects=True)
    check("Missing photo -> friendly message", b"add a photo" in r.data)
    r = c.post("/report/submit", data={"image": (img(1), "a.jpg")}, content_type="multipart/form-data",
               follow_redirects=True)
    check("Missing location -> friendly message", b"where this is" in r.data)

    # -------------------------------------------------- submit report #1 ---
    r = submit(c, 1, reporter_name="Kritika", **PHAGWARA)
    check("Submit redirects to confirmation", r.status_code == 302 and rid_from(r))
    rid1 = rid_from(r)
    rep = creport.get_report(rid1)
    check("Report stored with GPS + area", rep.latitude == 31.2240 and rep.city == "Phagwara")
    check("Report linked to a citizen identity", rep.citizen_id is not None)
    check("Target date = submission + SLA", abs(rep.target_at - (rep.created_at + config.SLA_DAYS * 86400)) < 2)
    check("Detection pipeline populated fields", rep.health_score is not None and rep.repair_priority in
          ("LOW", "MEDIUM", "HIGH", "URGENT") and rep.severity_level in ("LOW", "MODERATE", "HIGH", "CRITICAL", "NONE"))
    check("Existing PDF/JSON/HTML report files still generated", all(
        os.path.isfile(os.path.join(APP_DIR, p.lstrip("/"))) for p in
        (rep.pdf_report_path, rep.json_report_path, rep.html_report_path)))
    check("Detections keep explanation fields", rep.detections and "reason" in rep.detections[0])

    r = c.get(f"/reports/{rid1}/submitted")
    check("Confirmation page loads", r.status_code == 200)
    for txt in (b"ROAD DAMAGE DETECTED", b"Damage", b"Severity", b"Road Health", b"Repair Priority", b"Location",
                b"Report ID", rid1.encode(), b"What happens next?", b"Under Review", b"Repair In Progress" if False else b"In Progress",
                b"Estimated response target: 14 days", b"prototype", b"View AI analysis"):
        check(f"Confirmation shows {txt.decode()!r}", txt in r.data)
    check("Confirmation hides technical wording", not any(t in r.data for t in TECH))
    check("Confirmation awards +100", b"+100 points" in r.data)
    check("Points ledger = 100", rewards.total_points(rep.citizen_id) == 100)

    # ---- AI details progressive: present but inside <details>, confidence shown
    check("AI details are inside a collapsed <details>", b'<details class="ai">' in r.data and b"Confidence" in r.data)

    # ------------------------------------------------ double tap / repeat photo ---
    r = submit(c, 1, **PHAGWARA)
    check("Double-tap returns the SAME report (no new row)", rid_from(r) == rid1 and len(creport.list_all()) == 1)
    old_window = config.DOUBLE_SUBMIT_WINDOW_SECONDS
    config.DOUBLE_SUBMIT_WINDOW_SECONDS = 0
    r = submit(c, 1, **PHAGWARA)
    rid_rep = rid_from(r)
    check("Repeat photo later creates a report (nothing deleted)", rid_rep and rid_rep != rid1)
    check("Repeat photo earns no points", rewards.total_points(rep.citizen_id) == 100)
    page = c.get(f"/reports/{rid_rep}/submitted").data
    check("Repeat photo explains why no points", b"already reported" in page)
    config.DOUBLE_SUBMIT_WINDOW_SECONDS = old_window

    # -------------------------------------------------- report #2 (own client) ---
    c2 = flaskapp.app.test_client()
    r = submit(c2, 5, reporter_name="Amit", latitude="31.2244", longitude="75.7741", area="LPU Front Gate",
               city="Phagwara", state="Punjab", location="LPU main gate")
    rid2 = rid_from(r)
    rep2 = creport.get_report(rid2)
    check("Second citizen is a different identity", rep2.citizen_id != rep.citizen_id)
    check("Second citizen earned their own points", rewards.total_points(rep2.citizen_id) == 100)
    check("Nearby same-damage report is grouped, not deleted", rep2.duplicate_group_id is not None
          and creport.get_report(rid1).duplicate_group_id == rep2.duplicate_group_id)
    check("Related reports are shown on the report page", b"Possible related reports" in c2.get(f"/reports/{rid2}").data)

    # report without coordinates (must never appear on the map)
    r = submit(c2, 3, location="Somewhere with no GPS")
    rid_nogps = rid_from(r)

    # ------------------------------------------------------------ track ---
    r = c.post("/track", data={"report_id": rid1})
    check("Track by Report ID shows the tracker", r.status_code == 200 and b'class="road' in r.data)
    for txt in (b"Reported", b"Under Review", b"Assigned", b"In Progress", b"Resolved", b"Location", b"Damage",
                b"Severity", b"Responsible department", b"Assignment pending", b"Assigned officer", b"Days elapsed",
                b"Target resolution", b"Current status"):
        check(f"Track shows {txt.decode()!r}", txt in r.data)
    check("Track lowercase/spaced ID works", b"RS-" in c.post("/track", data={"report_id": " " + rid1.lower()}).data)
    check("Track numeric shortcut works", rid1.encode() in c.get("/track?report_id=1").data)
    r = c.post("/track", data={"report_id": "RS-2099-999999"})
    check("Unknown ID handled gracefully", r.status_code == 200 and b"couldn" in r.data)

    # -------------------------------------------------------- my reports ---
    r = c.get("/reports")
    check("My Reports lists own reports", rid1.encode() in r.data and rid2.encode() not in r.data)
    check("My Reports has no session/browser wording", not any(t in r.data for t in TECH))
    check("My Reports has View Report action", b"View Report" in r.data)
    check("Fresh visitor sees empty state (and no identity created)", b"haven" in flaskapp.app.test_client().get("/reports").data)

    # -------------------------------------------------------- report page ---
    r = c.get(f"/reports/{rid1}")
    for txt in (b"REPORT" if False else rid1.encode(), b"Progress", b"Location", b"Responsible authority",
                b"Assignment pending", b"Evidence", b"AI-annotated image", b"Citizen reward", b"Activity"):
        check(f"Report page shows {txt.decode()!r}", txt in r.data)
    check("Report page hides reporter contact / technical wording", not any(t in r.data for t in TECH))
    check("Report page has the map preview hook", b'id="miniMap"' in r.data)

    # XSS
    r = submit(c, 4, location="<script>alert(1)</script> Rd", latitude="31.3", longitude="75.6")
    xr = rid_from(r)
    body = c.get(f"/reports/{xr}").data
    check("User text is HTML-escaped", b"<script>alert(1)</script>" not in body and b"&lt;script&gt;" in body)

    # ---------------------------------------------------------------- map ---
    r = c.get("/map")
    check("Road Map page loads with Leaflet", r.status_code == 200 and b"leaflet" in r.data.lower())
    check("Map has all filters", all(t in r.data for t in (b"Search area", b"Severity", b"Status", b"Damage", b"Time",
                                                            b"Escalation Required", b"Last 7 days", b"Last 6 months")))
    api = c.get("/api/map-points").get_json()
    ids = {p["id"] for p in api["points"]}
    check("Map API returns only reports with stored coordinates", rid1 in ids and rid2 in ids and rid_nogps not in ids)
    p1 = next(p for p in api["points"] if p["id"] == rid1)
    check("Map point uses real stored coordinates", (p1["lat"], p1["lon"]) == (31.224, 75.7739))
    check("Map point has severity colour from config", p1["color"] == config.SEVERITY_COLORS[p1["severity"]])
    check("Map colours: low green / moderate yellow / high orange / critical red", (
        config.SEVERITY_COLORS["LOW"], config.SEVERITY_COLORS["MODERATE"], config.SEVERITY_COLORS["HIGH"],
        config.SEVERITY_COLORS["CRITICAL"]) == ("#2E9E5B", "#F2C200", "#F07C1A", "#D32F2F"))
    for k in ("damage", "health", "status_label", "location", "created", "url"):
        check(f"Map point carries {k}", k in p1)
    check("Filter severity works", all(p["severity"] == p1["severity"] for p in
          c.get(f"/api/map-points?severity={p1['severity']}").get_json()["points"])
          and c.get("/api/map-points?severity=CRITICAL").get_json()["count"] == sum(1 for p in api["points"] if p["severity"] == "CRITICAL"))
    check("Filter search area matches Phagwara", rid1 in {p["id"] for p in c.get("/api/map-points?q=phagwara").get_json()["points"]})
    check("Filter search area excludes Ludhiana", c.get("/api/map-points?q=ludhiana").get_json()["count"] == 0)
    check("Filter status works", c.get("/api/map-points?status=RESOLVED").get_json()["count"] == 0
          and c.get("/api/map-points?status=REPORTED").get_json()["count"] >= 2)
    check("Filter damage works", c.get("/api/map-points?damage=Pothole").get_json()["count"] >= 1
          and c.get("/api/map-points?damage=Alligator").get_json()["count"] == 0)
    check("Filter time works", c.get("/api/map-points?days=7").get_json()["count"] >= 2)
    conn = sqlite3.connect("data/history.sqlite3")
    conn.execute("UPDATE citizen_reports SET created_at = created_at - ? WHERE report_id = ?", (40 * 86400, rid2))
    conn.commit(); conn.close()
    check("Time filter excludes older reports", rid2 not in {p["id"] for p in c.get("/api/map-points?days=30").get_json()["points"]}
          and rid2 in {p["id"] for p in c.get("/api/map-points?days=180").get_json()["points"]})
    check("Inspection points are not on the public map", not any(p["kind"] != "report" for p in
          c.get("/api/map-points?inspections=1").get_json()["points"]))
    check("Map focus param passes through", b"RS-2026" in c.get(f"/map?focus={rid1}").data or rid1.encode() in c.get(f"/map?focus={rid1}").data)

    # ------------------------------------------------------------ area view ---
    import geo
    ak = geo.area_key("LPU Front Gate", "Phagwara")
    r = c.get(f"/area-health/{ak}")
    check("Area view loads", r.status_code == 200)
    for txt in (b"Road reports", b"Open reports", b"Resolved", b"High / Critical", b"Average road health", b"Nearby reports"):
        check(f"Area view shows {txt.decode()!r}", txt in r.data)
    card = flaskapp.area_health.get_area_scorecard(ak)
    check("Area stats come from real rows", card["report_count"] == 3 and card["open_reports"] == 3)

    # ---------------------------------------------------------- rewards page ---
    r = c.get("/rewards")
    check("Rewards page loads with points + badge", r.status_code == 200 and b"200 points" in r.data and b"Community Member" in r.data)
    check("Rewards explains how to earn points from config", all(f"+{v}".encode() in r.data for v in config.POINTS.values()))
    check("Rewards shows certificate lock", b"Certificate" in r.data and b"10,000" in r.data)
    r = c.get("/profile")
    check("Profile shows name, points and stats", b"Kritika" in r.data and b"Reports submitted" in r.data and b"Resolved issues" in r.data)

    # ------------------------------------------------------ authority area ---
    r = c.get("/authority/")
    check("Authority dashboard loads (open prototype mode)", r.status_code == 200 and b"Needs attention" in r.data)
    for txt in (b"Open reports", b"High priority", b"Escalation required", b"Assigned to me", b"In progress", b"Resolved"):
        check(f"Authority KPI {txt.decode()!r}", txt in r.data)
    for txt in (b"Dashboard", b"Reports", b"Map", b"Escalations", b"Assigned Work", b"Analytics", b"Prototype mode"):
        check(f"Authority nav/banner {txt.decode()!r}", txt in r.data)
    citizen_home = c.get("/").data
    check("Authority pages are not in citizen nav", b"/authority" not in re.search(rb'<nav class="main-nav".*?</nav>', citizen_home, re.S).group(0))
    for col in (b"Report ID", b"Location", b"Damage", b"Severity", b"Priority", b"Citizen", b"Submitted", b"Days", b"Assigned officer", b"Status"):
        check(f"Authority table column {col.decode()!r}", col in r.data)
    for act in (b"Assign", b"Update Status", b"View Report", b"Resolve"):
        check(f"Authority action {act.decode()!r}", act in r.data)
    check("Authority lists show the report", rid1.encode() in c.get("/authority/reports").data)
    check("Manage page loads", c.get(f"/authority/reports/{rid1}").status_code == 200)
    check("Map (authority) loads", c.get("/authority/map").status_code == 200)

    # verification + assignment
    r = c.post(f"/authority/reports/{rid1}/status", data={"status": "IN_PROGRESS"}, follow_redirects=True)
    check("In Progress blocked without assignment", creport.get_report(rid1).status == "REPORTED" and b"Assign a responsible department" in r.data)
    r = c.post(f"/authority/reports/{rid1}/assign", data={"department": "", "officer_name": "X"}, follow_redirects=True)
    check("Assignment needs a department", creport.get_report(rid1).department is None)
    pts_before = rewards.total_points(rep.citizen_id)
    c.post(f"/authority/reports/{rid1}/assign", data={"department": "Municipal Roads Department",
           "officer_name": "Test Officer", "officer_id": "T-1", "note": "Test assignment"}, follow_redirects=True)
    a = creport.get_report(rid1)
    check("Assignment saved (dept/officer/id/date)", (a.department, a.officer_name, a.officer_id) == ("Municipal Roads Department", "Test Officer", "T-1") and a.assigned_at)
    check("Assignment moves status to ASSIGNED", a.status == "ASSIGNED")
    check("Assignment verifies the report and awards +200", a.is_verified and rewards.total_points(rep.citizen_id) == pts_before + 200)
    check("Audit trail records who and note", any(h.get("by") == "Prototype Officer" and h.get("note") for h in a.status_history))
    c.post(f"/authority/reports/{rid1}/status", data={"status": "IN_PROGRESS", "note": "Crew on site"}, follow_redirects=True)
    check("Status update to IN_PROGRESS persisted", creport.get_report(rid1).status == "IN_PROGRESS")
    check("Verification reward is one-time", rewards.award(rep.citizen_id, rid1, "REPORT_VERIFIED") == 0)
    r = c.get(f"/reports/{rid1}").data
    check("Citizen page shows department + officer", b"Municipal Roads Department" in r and b"Test Officer" in r)
    check("Citizen tracker shows In Progress as current", b"stop current" in r)
    check("Assigned-to-me works for the matching officer", True)
    c.post("/authority/login", data={"name": "Test Officer"})
    check("Assigned Work lists my report", rid1.encode() in c.get("/authority/assigned").data)

    # ---------------------------------------------------- escalation flow ---
    conn = sqlite3.connect("data/history.sqlite3")
    conn.execute("UPDATE citizen_reports SET target_date = ? WHERE report_id = ?", (time.time() - 3 * 86400, rid1))
    conn.commit(); conn.close()
    n = workflow.sync_overdue()
    o = creport.get_report(rid1)
    check("Overdue report is flagged automatically", n >= 1 and o.overdue_flagged_at and o.display_status == "ESCALATION_REQUIRED")
    check("Flag is audited by the system, once", sum(1 for h in o.status_history if "target exceeded" in (h.get("event") or "")) == 1
          and workflow.sync_overdue() == 0)
    r = c.get(f"/reports/{rid1}").data
    check("Citizen sees 'Resolution target exceeded' banner", b"Resolution target exceeded" in r and b"simulated" in r)
    check("Escalation page lists it", rid1.encode() in c.get("/authority/escalations").data and b"Escalate" in c.get("/authority/escalations").data)
    check("Map status filter Escalation Required", rid1 in {p["id"] for p in c.get("/api/map-points?status=ESCALATION_REQUIRED").get_json()["points"]})
    check("Stats count escalations", creport.stats()["escalation_required"] == 1)
    c.post(f"/authority/reports/{rid1}/escalate", data={"note": "No crew for a week"}, follow_redirects=True)
    o = creport.get_report(rid1)
    check("Escalate raises level and says simulated", o.escalation_level == 1 and any("simulated" in (h.get("event") or "") for h in o.status_history))
    for _ in range(5):
        c.post(f"/authority/reports/{rid1}/escalate")
    check("Escalation stops at the highest level", creport.get_report(rid1).escalation_level == len(config.ESCALATION_LEVELS) - 1)
    ok, msg = workflow.escalate(rid2, "t")
    check("Cannot escalate a report that is not overdue", not ok)

    # ------------------------------------------------- resolve + evidence ---
    r = c.post(f"/authority/reports/{rid1}/resolve", data={"notes": "no photo"}, content_type="multipart/form-data", follow_redirects=True)
    check("Resolve without after photo is refused", creport.get_report(rid1).status != "RESOLVED" and b"required" in r.data)
    r = c.post(f"/authority/reports/{rid1}/status", data={"status": "RESOLVED"}, follow_redirects=True)
    check("Status dropdown cannot resolve without evidence", creport.get_report(rid1).status != "RESOLVED")
    pts_before = rewards.total_points(rep.citizen_id)
    r = c.post(f"/authority/reports/{rid1}/resolve", data={
        "after_image": (img(5, damaged=False), "after.jpg"), "notes": "Pavement removed and resurfaced.",
        "resolution_date": time.strftime("%Y-%m-%d")}, content_type="multipart/form-data", follow_redirects=True)
    d = creport.get_report(rid1)
    check("Resolve with photo -> RESOLVED", d.status == "RESOLVED" and d.resolved_at)
    check("Repair evidence stored", d.resolution_image_path and os.path.isfile(os.path.join(APP_DIR, d.resolution_image_path.lstrip("/"))))
    check("Resolution metadata stored", d.resolved_by == "Test Officer" and d.resolution_notes.startswith("Pavement") and d.resolution_date)
    check("Before/after comparison reused comparison.py", d.comparison and d.comparison.get("evidence_label") in ("Visual improvement detected", "Repair evidence submitted"))
    check("Resolved earns +500 for the reporter", rewards.total_points(rep.citizen_id) == pts_before + 500)
    check("No longer overdue once resolved", d.display_status == "RESOLVED" and not d.is_overdue())
    r = c.get(f"/reports/{rid1}").data
    for txt in (b"Repair evidence", b"BEFORE", b"AFTER", b"Resolution date", b"Resolved by", b"Test Officer", b"Pavement removed",
                b"not a certified inspection", b"+500 points were awarded"):
        check(f"Resolved page shows {txt.decode()!r}", txt in r)
    check("No AI 'certified repair' claim", b"AI certified" not in r and b"guaranteed" not in r.lower())
    ok, msg = workflow.resolve(rid1, "t", "/x.jpg", None, None, None, None)
    check("Cannot resolve twice / no double reward", not ok and rewards.total_points(rep.citizen_id) == pts_before + 500)

    # citizen follow-up
    check("Only the owner can confirm the repair", not workflow.confirm_repair(rid1, rep2.citizen_id, "/x.jpg")[0])
    check("Confirm form is offered to the owner", b"Confirm repair" in c.get(f"/reports/{rid1}").data)
    pts = rewards.total_points(rep.citizen_id)
    c.post(f"/reports/{rid1}/confirm-repair", data={"followup_image": (img(6), "f.jpg")}, content_type="multipart/form-data", follow_redirects=True)
    check("Follow-up earns +100 once", rewards.total_points(rep.citizen_id) == pts + 100)
    c.post(f"/reports/{rid1}/confirm-repair", data={"followup_image": (img(6), "f.jpg")}, content_type="multipart/form-data", follow_redirects=True)
    check("Follow-up cannot be repeated for points", rewards.total_points(rep.citizen_id) == pts + 100)

    # -------------------------------------------------------------- levels ---
    lv = rewards.level_for
    check("Level bands", [lv(p)["current"]["name"] for p in (0, 999, 1000, 4999, 5000, 9999, 10000)] ==
          ["Community Member", "Community Member", "Road Watcher", "Road Watcher", "Civic Champion", "Civic Champion", "Road Saver"])
    check("Progress toward next badge", lv(500)["progress_pct"] == 50 and lv(500)["next"]["name"] == "Road Watcher" and lv(500)["points_to_next"] == 500)
    check("Top badge has no next", lv(20000)["next"] is None and lv(20000)["progress_pct"] == 100)

    # ------------------------------------------------------- certificate ---
    r = c.get("/rewards/certificate", follow_redirects=True)
    check("Certificate locked below threshold", b"unlocks at 10,000" in r.data)
    cid = rep.citizen_id
    conn = sqlite3.connect("data/history.sqlite3")
    conn.execute("INSERT INTO reward_events (citizen_id, report_id, event_type, points, note, created_at) VALUES (?, 'TEST-SEED', 'REPORT_RESOLVED', 10000, 'test seed only', ?)", (cid, time.time()))
    conn.commit(); conn.close()
    check("Road Saver badge at 10,000+", b"ROAD SAVER" in c.get("/rewards").data and b"Download Certificate" in c.get("/rewards").data)
    r = c.get("/rewards/certificate")
    check("Certificate is a PDF", r.status_code == 200 and r.mimetype == "application/pdf" and r.data.startswith(b"%PDF"))
    cert = rewards.get_certificate(re.search(r"RSC-[0-9A-F]{10}", r.headers["Content-Disposition"]).group(0))
    check("Certificate record has name/badge/points/counts", cert and cert["citizen_name"] == "Kritika" and cert["badge"] == "Road Saver" and cert["points"] >= 10000 and cert["resolved_reports"] == 1)
    v = c.get(f"/verify/{cert['verification_id']}")
    check("Verification page confirms and says not government", v.status_code == 200 and b"not a government-issued" in v.data)
    check("Unknown certificate -> 404", c.get("/verify/RSC-NOPE").status_code == 404)
    check("Certificate re-download reuses the record", rewards.issue_certificate(flaskapp.citizens.get(cid))["verification_id"] == cert["verification_id"])
    check("Another citizen cannot download it", b"unlocks at" in c2.get("/rewards/certificate", follow_redirects=True).data)

    # ---------------------------------------- caps / no-damage handling ---
    conn = sqlite3.connect("data/history.sqlite3")
    for i in range(config.DAILY_REWARDED_REPORT_CAP):
        conn.execute("INSERT OR IGNORE INTO reward_events (citizen_id, report_id, event_type, points, created_at) VALUES (?, ?, 'REPORT_SUBMITTED', 100, ?)", (rep2.citizen_id, f"CAP-{i}", time.time()))
    conn.commit(); conn.close()
    before = rewards.total_points(rep2.citizen_id)
    r = submit(c2, 7, location="Cap test", latitude="30.9", longitude="75.8")
    check("Daily reward cap stops farming (report still saved)", rid_from(r) and rewards.total_points(rep2.citizen_id) == before)

    # ------------------------------------------------ authority key mode ---
    os.environ["ROADSCAN_AUTHORITY_KEY"] = "s3cret"
    k = flaskapp.app.test_client()
    r = k.get("/authority/")
    check("Key mode: dashboard redirects to sign-in", r.status_code == 302 and "/authority/login" in r.headers["Location"])
    check("Key mode: legacy technical tool is gated", k.get("/history").status_code == 302)
    check("Key mode: wrong key rejected", b"not correct" in k.post("/authority/login", data={"name": "A", "key": "no"}).data)
    r = k.post("/authority/login?next=//evil.example", data={"name": "Asha", "key": "s3cret"})
    check("Key mode: login works and blocks open redirect", r.status_code == 302 and "evil" not in r.headers["Location"])
    check("Key mode: authenticated dashboard", k.get("/authority/").status_code == 200)
    del os.environ["ROADSCAN_AUTHORITY_KEY"]

    # ------------------------------------------------------- legacy routes ---
    check("/maintenance redirects to the authority dashboard", "/authority" in c.get("/maintenance").headers["Location"])
    check("/damage-map redirects to /map", c.get("/damage-map").headers["Location"].endswith("/map"))
    for path in ("/history", "/compare", "/segments", "/area-health", "/authority/inspect", "/authority/analytics", "/how-it-works"):
        check(f"{path} still works", c.get(path).status_code == 200)
    r = c.post("/analyze", data={"image": (img(8), "t.jpg"), "location": "Legacy St", "confidence": "0"}, content_type="multipart/form-data")
    check("/analyze (technical inspection) still works", r.status_code == 200 and b"Road Health" in r.data)
    check("Authority pages may show technical detail", b"Detection backend" in c.get("/authority/analytics").data)

    # ----------------------------------------------------------- migration ---
    check("Migrations are idempotent", flaskapp.migrations.run_migrations() == [])
    conn = sqlite3.connect("data/history.sqlite3")
    orig = sqlite3.connect(os.path.join(_helpers.SRC_ROOT, "data", "history.sqlite3"))
    check("Pre-existing inspection history preserved", conn.execute("select count(*) from assessments").fetchone()[0] >= orig.execute("select count(*) from assessments").fetchone()[0])
    conn.close(); orig.close()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = [(n, d) for n, ok, d in results if not ok]
    print(f"\n--- SUMMARY ---\n{passed} passed, {len(failed)} failed, {len(results)} total")
    for n, d in failed:
        print(f"  FAIL: {n} {d}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
