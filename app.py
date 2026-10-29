"""
app.py
------
MODULE: Web Application / Orchestration (Flask)
INPUT:  HTTP requests (image uploads, form fields)
OUTPUT: rendered HTML pages + generated report files
WHY:    ties every other module together into the end-to-end workflow:

    ROAD IMAGE -> detector.py -> segmentation (masks on Detection)
        -> severity.py -> health_score.py -> severity.compute_repair_priority
        -> history.py / comparison.py -> report.py -> templates/*.html

This file contains no detection, scoring, or comparison LOGIC itself - it
only reads uploads, calls the pipeline functions in order, and renders
results. That separation is what makes each module independently
explainable (see README "Module-by-module explanation").

Uploaded-image workflow only: no live camera, no streaming, no real-time
driver alerts.
"""

import hashlib
import os
import time
import uuid
from datetime import timedelta

import cv2
import numpy as np
from flask import (Flask, render_template, request, url_for, redirect, flash, session,
                   jsonify, Response)
from io import BytesIO
from xml.sax.saxutils import escape

from flask import send_file
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)

from detector import get_default_detector, DetectorUnavailableError
import severity as sev
import health_score as hsc
import explain
import history
import dedup
import comparison as cmp
import report as rpt
import citizen_report as creport
import geo
import migrations
import area_health
import segment as seg
import damage_map
import config
import citizens
import rewards
import workflow
from authority_views import authority_required, register_authority

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
RESULT_DIR = os.path.join(APP_ROOT, "static", "results")
REPORT_DIR = os.path.join(APP_ROOT, "static", "reports")
TEMPLATES_DIR = os.path.join(APP_ROOT, "templates")
for d in (UPLOAD_DIR, RESULT_DIR, REPORT_DIR, os.path.join(APP_ROOT, "data")):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}   # the citizen UI advertises JPG/PNG only

app = Flask(__name__)
# Override in any real deployment: set ROADSCAN_SECRET_KEY in the environment.
app.secret_key = os.environ.get("ROADSCAN_SECRET_KEY", "road-damage-assessment-dev")
app.permanent_session_lifetime = timedelta(days=365)

# --- Schema migration for the GPS/Area/Segment/Duplicate-grouping columns
# added by the location-aware enhancement, run once at startup so a fresh
# install AND an existing database both end up with the columns/tables
# those features need. Additive only - see migrations.py docstring. ---
_MIGRATION_CHANGES = migrations.run_migrations()
if _MIGRATION_CHANGES:
    print(f"[Roadscan] Database migrated - added: {', '.join(_MIGRATION_CHANGES)}")
else:
    print("[Roadscan] Database schema already up to date.")

# --- Detector selection happens ONCE at startup, and the result (which
# backend actually loaded) is kept and shown everywhere in the UI. ---
DETECTOR_STATUS = get_default_detector()
DETECTOR = DETECTOR_STATUS.detector
print(f"[Roadscan] {DETECTOR_STATUS.message}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    max_side = 1600  # keeps inference fast on a normal laptop
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def run_pipeline(image_bgr: np.ndarray, min_confidence: float = 0.0):
    """
    Shared detect -> segment -> severity -> health -> priority pipeline.
    `min_confidence` filters detections after the fact (works identically
    regardless of which backend produced them).
    """
    detections = DETECTOR.detect(image_bgr)
    if min_confidence > 0:
        detections = [d for d in detections if d.confidence >= min_confidence]
    assessments = sev.assess_damages(detections, image_bgr.shape)
    health = hsc.compute_health_score(assessments, image_bgr.shape)
    priority = sev.compute_repair_priority(assessments, health)
    total_area_ratio = sum(a.area_ratio for a in assessments)
    return detections, assessments, health, priority, total_area_ratio


def _parse_optional_float(raw):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_location_fields(form) -> dict:
    """
    Shared GPS + manual-area parsing for BOTH Inspect Road (/analyze) and
    Report Road Damage (/report/submit) - see PART 1 of the enhancement
    spec. GPS is entirely optional; every field falls back to None rather
    than a fabricated value.
    """
    latitude = _parse_optional_float(form.get("latitude"))
    longitude = _parse_optional_float(form.get("longitude"))
    area = (form.get("area") or "").strip() or None
    city = (form.get("city") or "").strip() or None
    state = (form.get("state") or "").strip() or None
    road_name = (form.get("road_name") or "").strip() or None
    landmark = (form.get("landmark") or "").strip() or None
    return {
        "latitude": latitude, "longitude": longitude,
        "area": area, "city": city, "state": state,
        "road_name": road_name, "landmark": landmark,
    }


def save_upload(file_storage) -> str:
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    file_storage.save(path)
    return path


def _derive_location(loc: dict, typed: str) -> str:
    """Human-readable location text: what the citizen typed, else built from the parts they gave."""
    typed = (typed or "").strip()
    if typed:
        return typed
    parts = [loc["landmark"], loc["road_name"], loc["area"], loc["city"], loc["state"]]
    text = ", ".join(p for p in parts if p)
    if not text and loc["latitude"] is not None and loc["longitude"] is not None:
        text = f"GPS location ({loc['latitude']:.5f}, {loc['longitude']:.5f})"
    return text


@app.context_processor
def inject_globals():
    cz = citizens.current_citizen(session)
    return {
        # technical details - only rendered inside the authority layout
        "detector_name": DETECTOR.name,
        "detector_is_primary": DETECTOR_STATUS.is_primary,
        "detector_message": DETECTOR_STATUS.message,
        # citizen identity + product constants for templates
        "cz": cz,
        "cz_points": rewards.total_points(cz.id) if cz else 0,
        "cfg": config,
        "status_labels": creport.STATUS_LABELS,
    }


@app.template_filter("fmtdate")
def fmtdate(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


@app.template_filter("fmtday")
def fmtday(ts):
    return time.strftime("%d %b %Y", time.localtime(ts)) if ts else "-"


@app.template_filter("fmtnum")
def fmtnum(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return n


_last_overdue_sync = 0.0


@app.before_request
def _refresh_overdue_flags():
    """Mark reports past their resolution target (at most once a minute; cheap)."""
    global _last_overdue_sync
    if request.endpoint == "static":
        return
    now = time.time()
    if now - _last_overdue_sync > 60:
        _last_overdue_sync = now
        try:
            workflow.sync_overdue()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    s = creport.stats()
    impact = {
        "reports": s["total"],
        "resolved": s["resolved"],
        "areas": len(area_health.list_areas()),   # areas with at least one real observation
    }
    return render_template("index.html", impact=impact)


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/analyze", methods=["POST"])
@authority_required
def analyze():
    file = request.files.get("image")
    location = (request.form.get("location") or "").strip()
    loc = _parse_location_fields(request.form)
    try:
        min_conf = float(request.form.get("confidence", 0)) / 100.0
    except ValueError:
        min_conf = 0.0

    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please upload a valid image file (png/jpg/jpeg/webp/bmp).")
        return redirect(url_for("authority.inspect"))

    original_name = file.filename
    upload_path = save_upload(file)
    image_bgr = read_image(upload_path)

    try:
        detections, assessments, health, priority, total_area_ratio = run_pipeline(image_bgr, min_conf)
    except DetectorUnavailableError as e:
        flash(f"Detector error: {e}")
        return redirect(url_for("authority.inspect"))

    severities_by_id = {id(a.detection): a.severity for a in assessments}
    annotated = explain.draw_annotations(image_bgr, detections, severities_by_id)
    heatmap = explain.evidence_heatmap(image_bgr, detections)

    result_id = uuid.uuid4().hex
    annotated_path = os.path.join(RESULT_DIR, f"{result_id}_annotated.jpg")
    heatmap_path = os.path.join(RESULT_DIR, f"{result_id}_heatmap.jpg")
    cv2.imwrite(annotated_path, annotated)
    cv2.imwrite(heatmap_path, heatmap)

    history_comparison = None
    duplicate_note = None
    if location:
        prior_records = history.get_history(location)
        if prior_records:
            try:
                prev = prior_records[-1]
                prev_img_path = os.path.join(APP_ROOT, prev.image_path.lstrip("/"))
                prev_img = cv2.imread(prev_img_path)
                if prev_img is not None:
                    prev_dets, prev_assess, _, _, _ = run_pipeline(prev_img, min_conf)
                    prev_sigs = dedup.compute_signatures(prev_img, prev_dets, prev.image_name)
                    new_sigs = dedup.compute_signatures(image_bgr, detections, original_name)
                    matches = dedup.match_against_history(new_sigs, prev_sigs)
                    new_d, repeat_d = dedup.deduplicate_counts(detections, matches)
                    if repeat_d:
                        duplicate_note = (
                            f"{len(repeat_d)} of {len(detections)} detections in this image closely "
                            f"match damage seen in the previous assessment of '{location}' (likely the "
                            f"same physical damage, not new); {len(new_d)} appear to be new."
                        )
            except Exception:
                duplicate_note = None  # dedup is best-effort; never block the main pipeline on it
        segment_id = geo.make_segment_id(loc["road_name"], loc["area"], loc["city"],
                                          fallback_location=location)
        history.save_assessment(
            location=location, image_name=original_name,
            image_path=url_for("static", filename=f"uploads/{os.path.basename(upload_path)}"),
            annotated_path=url_for("static", filename=f"results/{result_id}_annotated.jpg"),
            health_score=health.score, condition=health.condition,
            damage_count=len(assessments), priority=priority.level,
            damaged_area_ratio=total_area_ratio,
            detections_summary=[{
                "class_id": a.detection.class_id, "class_name": a.detection.class_name,
                "severity": a.severity, "confidence": round(a.detection.confidence, 3),
            } for a in assessments],
            latitude=loc["latitude"], longitude=loc["longitude"],
            area=loc["area"], city=loc["city"], state=loc["state"],
            road_name=loc["road_name"], landmark=loc["landmark"], segment_id=segment_id,
        )
        history_comparison = history.compare_latest_two(location)
        area_key = geo.area_key(loc["area"], loc["city"])
        if area_key:
            session["last_area_key"] = area_key

    report_dict = rpt.build_report_dict(
        image_name=original_name, assessments=assessments, health=health, priority=priority,
        location=location or None, detector_backend=DETECTOR.name,
        is_primary_detector=DETECTOR_STATUS.is_primary, history_comparison=history_comparison,
    )
    json_path = os.path.join(REPORT_DIR, f"{result_id}_report.json")
    pdf_path = os.path.join(REPORT_DIR, f"{result_id}_report.pdf")
    html_path = os.path.join(REPORT_DIR, f"{result_id}_report.html")
    rpt.save_json_report(report_dict, json_path)
    rpt.save_pdf_report(report_dict, annotated_path, pdf_path)
    rpt.save_html_report(report_dict, url_for("static", filename=f"results/{result_id}_annotated.jpg"),
                          TEMPLATES_DIR, html_path)

    # damage-type distribution + severity distribution for the dashboard cards
    type_counts = {}
    severity_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for a in assessments:
        type_counts[a.detection.class_name] = type_counts.get(a.detection.class_name, 0) + 1
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

    return render_template(
        "results.html",
        image_name=original_name,
        location=location,
        annotated_url=url_for("static", filename=f"results/{result_id}_annotated.jpg"),
        heatmap_url=url_for("static", filename=f"results/{result_id}_heatmap.jpg"),
        assessments=assessments,
        health=health,
        priority=priority,
        total_area_ratio=total_area_ratio,
        type_counts=type_counts,
        severity_counts=severity_counts,
        json_url=url_for("static", filename=f"reports/{result_id}_report.json"),
        pdf_url=url_for("static", filename=f"reports/{result_id}_report.pdf"),
        html_url=url_for("static", filename=f"reports/{result_id}_report.html"),
        history_comparison=history_comparison,
        duplicate_note=duplicate_note,
        inspection_id=report_dict["inspection_id"],
    )


@app.route("/history")
@authority_required
def history_list():
    locations = history.list_locations()
    summaries = []
    for loc in locations:
        records = history.get_history(loc)
        latest = records[-1]
        comparison = history.compare_latest_two(loc)
        summaries.append({
            "location": loc,
            "num_assessments": len(records),
            "latest_score": latest.health_score,
            "latest_condition": latest.condition,
            "latest_date": time.strftime("%Y-%m-%d", time.localtime(latest.created_at)),
            "trend_status": comparison["trend_status"] if comparison else "insufficient data",
        })
    return render_template("history.html", summaries=summaries)


@app.route("/history/<path:location>")
@authority_required
def history_detail(location):
    records = history.get_history(location)
    comparison = history.compare_latest_two(location)

    chart_w, chart_h, pad = 600, 180, 20
    scores = [r.health_score for r in records]
    svg_points = ""
    svg_dots = []
    if len(records) >= 1:
        n = len(records)
        x_step = (chart_w - 2 * pad) / max(1, n - 1)
        coords = []
        for i, r in enumerate(records):
            x = pad + i * x_step
            y = pad + (chart_h - 2 * pad) * (1 - r.health_score / 100.0)
            coords.append((round(x, 1), round(y, 1)))
        svg_points = " ".join(f"{x},{y}" for x, y in coords)
        svg_dots = [{"x": x, "y": y, "score": r.health_score,
                     "date": time.strftime("%m/%d", time.localtime(r.created_at))}
                    for (x, y), r in zip(coords, records)]

    return render_template("history_detail.html", location=location, records=records,
                            comparison=comparison, svg_points=svg_points, svg_dots=svg_dots,
                            chart_w=chart_w, chart_h=chart_h)


@app.route("/compare", methods=["GET", "POST"])
@authority_required
def compare_page():
    if request.method == "GET":
        return render_template("compare.html")

    before_file = request.files.get("before_image")
    after_file = request.files.get("after_image")
    if not before_file or not after_file or before_file.filename == "" or after_file.filename == "":
        flash("Please upload both a before-repair and an after-repair image.")
        return redirect(url_for("compare_page"))
    if not (allowed_file(before_file.filename) and allowed_file(after_file.filename)):
        flash("Both files must be valid images (png/jpg/jpeg/webp/bmp).")
        return redirect(url_for("compare_page"))

    before_path = save_upload(before_file)
    after_path = save_upload(after_file)
    img_before = read_image(before_path)
    img_after = read_image(after_path)

    try:
        b_det, b_assess, b_health, b_priority, b_area = run_pipeline(img_before)
        a_det, a_assess, a_health, a_priority, a_area = run_pipeline(img_after)
    except DetectorUnavailableError as e:
        flash(f"Detector error: {e}")
        return redirect(url_for("compare_page"))

    result = cmp.compare_before_after(
        before_health=b_health.score, before_count=len(b_assess), before_area_ratio=b_area,
        after_health=a_health.score, after_count=len(a_assess), after_area_ratio=a_area,
        img_before=img_before, img_after=img_after,
    )

    b_ann = explain.draw_annotations(img_before, b_det, {id(a.detection): a.severity for a in b_assess})
    a_ann = explain.draw_annotations(img_after, a_det, {id(a.detection): a.severity for a in a_assess})
    rid = uuid.uuid4().hex
    b_ann_path = os.path.join(RESULT_DIR, f"{rid}_before.jpg")
    a_ann_path = os.path.join(RESULT_DIR, f"{rid}_after.jpg")
    cv2.imwrite(b_ann_path, b_ann)
    cv2.imwrite(a_ann_path, a_ann)

    return render_template(
        "compare_results.html",
        before_url=url_for("static", filename=f"results/{rid}_before.jpg"),
        after_url=url_for("static", filename=f"results/{rid}_after.jpg"),
        before_health=b_health, after_health=a_health,
        before_priority=b_priority, after_priority=a_priority,
        before_count=len(b_assess), after_count=len(a_assess),
        result=result,
    )


# ---------------------------------------------------------------------------
# Citizen road-reporting journey: REPORT -> TRACK -> REPAIR -> VERIFY -> REWARD
#
# Reuses the SAME detect -> severity -> health -> priority pipeline as
# /analyze (run_pipeline). No second detector or scoring engine. Identity
# comes from citizens.current_citizen() - the one seam a future login system
# would replace. Everything that happens to a report after submission goes
# through workflow.py so the audit trail and rewards stay consistent.
# ---------------------------------------------------------------------------

def _worst_assessment_severity(assessments):
    if not assessments:
        return "NONE", 0.0
    worst = max(assessments, key=lambda a: a.severity_score)
    return worst.severity, worst.severity_score


def _damage_summary_text(assessments) -> str:
    if not assessments:
        return "No damage detected"
    counts = {}
    for a in assessments:
        counts[a.detection.class_name] = counts.get(a.detection.class_name, 0) + 1
    return ", ".join(f"{name} x{n}" if n > 1 else name for name, n in counts.items())


_build_timeline = workflow.build_timeline


def _related_reports(rep):
    if not rep.duplicate_group_id:
        return []
    return [m for m in creport.list_by_group(rep.duplicate_group_id) if m.report_id != rep.report_id]


def _is_owner(rep) -> bool:
    cz = citizens.current_citizen(session)
    return bool(cz and rep.citizen_id and rep.citizen_id == cz.id)


@app.route("/report")
def report_form():
    return render_template("report_form.html")


@app.route("/report/submit", methods=["POST"])
def report_submit():
    file = request.files.get("image")
    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please add a photo of the road damage (JPG or PNG).")
        return redirect(url_for("report_form"))

    loc = _parse_location_fields(request.form)
    location = _derive_location(loc, request.form.get("location"))
    if not location:
        flash("Please tell us where this is - use your location or type it in.")
        return redirect(url_for("report_form"))

    road_name, landmark = loc["road_name"], loc["landmark"]
    latitude, longitude = loc["latitude"], loc["longitude"]
    description = (request.form.get("description") or "").strip()[:1000] or None
    reporter_name = (request.form.get("reporter_name") or "").strip()[:60] or None
    reporter_contact = (request.form.get("reporter_contact") or "").strip()[:120] or None

    data = file.read()
    file.stream.seek(0)
    image_hash = hashlib.sha256(data).hexdigest()

    cz = citizens.current_citizen(session, create_if_missing=True)
    if reporter_name and not cz.name:
        citizens.set_name(cz.id, reporter_name)

    # Double-tap guard: the same person sending the identical photo moments
    # ago gets their existing report back instead of a second one.
    for prior in reversed(creport.find_by_hash(image_hash)):
        if prior.citizen_id == cz.id and time.time() - prior.created_at < config.DOUBLE_SUBMIT_WINDOW_SECONDS:
            return redirect(url_for("report_submitted", report_id=prior.report_id))

    original_name = file.filename
    upload_path = save_upload(file)
    try:
        image_bgr = read_image(upload_path)
    except ValueError:
        flash("We couldn't read that photo. Please try a different JPG or PNG.")
        return redirect(url_for("report_form"))

    try:
        detections, assessments, health, priority, _total_area_ratio = run_pipeline(image_bgr)
    except DetectorUnavailableError:
        flash("Our damage analysis is temporarily unavailable. Please try again in a few minutes.")
        return redirect(url_for("report_form"))

    severities_by_id = {id(a.detection): a.severity for a in assessments}
    annotated = explain.draw_annotations(image_bgr, detections, severities_by_id)
    result_id = uuid.uuid4().hex
    annotated_path = os.path.join(RESULT_DIR, f"{result_id}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    severity_level, severity_score = _worst_assessment_severity(assessments)
    detections_summary = [{
        "class_id": a.detection.class_id, "class_name": a.detection.class_name,
        "confidence": round(a.detection.confidence, 3), "severity": a.severity,
        "bbox": a.detection.bbox, "area_ratio": round(a.area_ratio, 4),
        "severity_score": round(a.severity_score, 1), "reason": a.reason,
    } for a in assessments]

    # Same report-generation module (report.py) as /analyze - PDF/HTML/JSON are preserved.
    report_dict = rpt.build_report_dict(
        image_name=original_name, assessments=assessments, health=health, priority=priority,
        location=location, detector_backend=DETECTOR.name, is_primary_detector=DETECTOR_STATUS.is_primary,
    )
    json_path = os.path.join(REPORT_DIR, f"{result_id}_report.json")
    pdf_path = os.path.join(REPORT_DIR, f"{result_id}_report.pdf")
    html_path = os.path.join(REPORT_DIR, f"{result_id}_report.html")
    rpt.save_json_report(report_dict, json_path)
    rpt.save_pdf_report(report_dict, annotated_path, pdf_path)
    rpt.save_html_report(report_dict, url_for("static", filename=f"results/{result_id}_annotated.jpg"),
                          TEMPLATES_DIR, html_path)

    segment_id = geo.make_segment_id(road_name, loc["area"], loc["city"], fallback_location=location)
    report_id = creport.create_report(
        image_path=url_for("static", filename=f"uploads/{os.path.basename(upload_path)}"),
        annotated_image_path=url_for("static", filename=f"results/{result_id}_annotated.jpg"),
        latitude=latitude, longitude=longitude, location=location,
        road_name=road_name, landmark=landmark, description=description,
        reporter_name=reporter_name or cz.name, reporter_contact=reporter_contact,
        detected_damage=_damage_summary_text(assessments),
        detections_summary=detections_summary,
        severity_score=severity_score, severity_level=severity_level,
        health_score=health.score, health_condition=health.condition,
        repair_priority=priority.level, priority_reason=priority.reason,
        json_report_path=url_for("static", filename=f"reports/{result_id}_report.json"),
        pdf_report_path=url_for("static", filename=f"reports/{result_id}_report.pdf"),
        html_report_path=url_for("static", filename=f"reports/{result_id}_report.html"),
        area=loc["area"], city=loc["city"], state=loc["state"], segment_id=segment_id,
        citizen_id=cz.id, image_hash=image_hash,
    )

    # Best-effort duplicate grouping - never blocks submission.
    try:
        dedup.recompute_duplicate_groups()
    except Exception:
        pass

    workflow.award_submission(report_id)

    area_key = geo.area_key(loc["area"], loc["city"])
    if area_key:
        session["last_area_key"] = area_key
    return redirect(url_for("report_submitted", report_id=report_id))


@app.route("/reports/<report_id>/submitted")
def report_submitted(report_id):
    rep = creport.get_report(report_id)
    if not rep:
        flash("We couldn't find that report.")
        return redirect(url_for("report_form"))
    earned = rewards.has_award(rep.report_id, "REPORT_SUBMITTED")
    if earned:
        reason = None
    elif not rep.detections or rep.severity_level in (None, "NONE"):
        reason = "no_damage"
    else:
        reason = workflow.submission_block_reason(rep) or "pending"
    return render_template("report_submitted.html", rep=rep, timeline=_build_timeline(rep),
                            points_earned=config.POINTS["REPORT_SUBMITTED"] if earned else 0,
                            no_points_reason=reason, related=_related_reports(rep),
                            existing_photo=[r for r in creport.find_by_hash(rep.image_hash)
                                             if r.id < rep.id][:1])


@app.route("/reports/<report_id>")
def report_detail(report_id):
    rep = creport.get_report(report_id)
    if not rep:
        flash(f"We couldn't find a report with ID '{report_id}'.")
        return redirect(url_for("track_report"))
    owner = _is_owner(rep)
    resolved_reward = rewards.has_award(rep.report_id, "REPORT_RESOLVED")
    return render_template("report_detail.html", rep=rep, timeline=_build_timeline(rep),
                            related=_related_reports(rep), is_owner=owner,
                            resolved_reward=resolved_reward,
                            can_confirm=owner and rep.status == "RESOLVED" and not rep.citizen_followup_path)
@app.route("/reports/<report_id>/pdf")
def download_report_pdf(report_id):
    rep = creport.get_report(report_id)

    if not rep:
        flash(f"We couldn't find a report with ID '{report_id}'.")
        return redirect(url_for("track_report"))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"RoadScan Report {rep.report_id}",
        author="RoadScan",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#174a6e"),
        spaceAfter=6 * mm,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#174a6e"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    ))

    def safe(value):
        return escape(str(value if value not in (None, "") else "Not recorded"))


    def add_image(story, relative_path, caption):
        if not relative_path:
            return

        # Convert a web path such as /static/uploads/photo.jpg
        # into a path relative to the RoadScan application folder.
        raw = str(relative_path).replace("\\", "/").split("?", 1)[0]
        raw = raw.lstrip("/")

        if not raw.startswith("static/"):
            app.logger.warning("Skipping unexpected PDF image path: %s", raw)
            return

        static_root = os.path.abspath(os.path.join(APP_ROOT, "static"))
        path = os.path.abspath(
            os.path.join(APP_ROOT, *raw.split("/"))
        )

        # Keep image access inside RoadScan's static directory.
        if os.path.commonpath([static_root, path]) != static_root:
            app.logger.warning("PDF image path is outside static directory")
            return

        if not os.path.isfile(path):
            app.logger.warning("PDF image file not found: %s", path)
            return

        try:
            story.append(Paragraph(escape(caption), styles["Normal"]))
            image = Image(
                path,
                width=150 * mm,
                height=95 * mm,
                kind="proportional",
            )
            image.hAlign = "LEFT"
            story.append(image)
            story.append(Spacer(1, 3 * mm))
        except (OSError, ValueError):
            app.logger.exception("Could not add image to PDF: %s", path)


    story = [
        Paragraph("RoadScan", styles["ReportTitle"]),
        Paragraph("Road Damage Inspection Report", styles["Heading2"]),
        Spacer(1, 3 * mm),
    ]

    rows = [
        ["Report ID", safe(rep.report_id)],
        ["Submitted", safe(time.strftime("%d %b %Y, %H:%M", time.localtime(rep.created_at)))],
        ["Status", safe(rep.display_status_label)],
        ["Location", safe(rep.location)],
        ["Road", safe(rep.road_name)],
        ["Landmark", safe(rep.landmark)],
        ["Detected damage", safe(rep.detected_damage)],
        ["Severity", safe(rep.severity_level)],
        ["Severity score", safe(rep.severity_score)],
        ["Road health", f"{safe(rep.health_score)} / 100"],
        ["Health condition", safe(rep.health_condition)],
        ["Repair priority", safe(rep.repair_priority)],
        ["Priority reason", safe(rep.priority_reason)],
    ]

    table = Table(rows, colWidths=[43 * mm, 125 * mm], repeatRows=0)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf3f7")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#174a6e")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dee5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    if rep.description:
        story.extend([
            Spacer(1, 4 * mm),
            Paragraph("Citizen description", styles["SectionHeading"]),
            Paragraph(safe(rep.description), styles["Normal"]),
        ])

    story.append(Paragraph("Evidence images", styles["SectionHeading"]))
    add_image(story, rep.image_path, "Original citizen photo")
    add_image(story, rep.annotated_image_path, "Annotated damage image")

    if rep.resolution_image_path:
        add_image(story, rep.resolution_image_path, "Repair evidence photo")

    if rep.detections:
        story.append(Paragraph("Detection details", styles["SectionHeading"]))
        detection_rows = [["Damage type", "Confidence"]]
        for detection in rep.detections:
            if isinstance(detection, dict):
                detection_rows.append([
                    safe(detection.get("class_name") or detection.get("class_id") or "Damage"),
                    safe(detection.get("confidence", "Not recorded")),
                ])
        if len(detection_rows) > 1:
            details = Table(detection_rows, colWidths=[110 * mm, 58 * mm], repeatRows=1)
            details.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174a6e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dee5")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(details)

    story.extend([
        Spacer(1, 8 * mm),
        Paragraph(
            "Disclaimer: RoadScan provides a visual, AI-assisted assessment. "
            "This report is not a certified engineering inspection. "
            "Workflow status is recorded within the RoadScan prototype.",
            styles["Italic"],
        ),
    ])

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"RoadScan_{rep.report_id}.pdf",
    )

@app.route("/reports/<report_id>/confirm-repair", methods=["POST"])
def confirm_repair(report_id):
    cz = citizens.current_citizen(session)
    file = request.files.get("followup_image")
    if not cz:
        flash("Only the person who reported this can confirm the repair.")
    elif not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please add a JPG or PNG photo of the repaired road.")
    else:
        path = url_for("static", filename=f"uploads/{os.path.basename(save_upload(file))}")
        ok, msg = workflow.confirm_repair(report_id, cz.id, path)
        flash(msg)
    return redirect(url_for("report_detail", report_id=report_id))


def _normalise_report_id(raw: str) -> str:
    raw = (raw or "").strip().upper().replace(" ", "")
    if raw.isdigit():                       # "2" -> RS-<this year>-000002
        return f"RS-{time.strftime('%Y')}-{int(raw):06d}"
    return raw


@app.route("/track", methods=["GET", "POST"])
def track_report():
    raw = request.form.get("report_id") if request.method == "POST" else request.args.get("report_id")
    if not (raw or "").strip():
        if request.method == "POST":
            flash("Please enter a Report ID.")
        return render_template("track_report.html")
    rid = _normalise_report_id(raw)
    rep = creport.get_report(rid)
    if not rep:
        flash(f"We couldn't find a report with ID '{raw.strip()}'. Check the ID and try again.")
        return render_template("track_report.html", searched_id=raw.strip())
    return render_template("track_report.html", rep=rep, timeline=_build_timeline(rep), searched_id=rid)


@app.route("/reports")
def my_reports():
    cz = citizens.current_citizen(session)
    reports = creport.list_by_citizen(cz.id) if cz else []
    return render_template("my_reports.html", reports=reports)


# ---------------------------------------------------------------------------
# Road Map (Leaflet) - real stored coordinates only
# ---------------------------------------------------------------------------

@app.route("/map")
def road_map():
    return render_template("road_map.html", layout="base.html", authority=False,
                            focus=request.args.get("focus", ""))


@app.route("/damage-map")
def damage_map_page():
    return redirect(url_for("road_map"), code=301)


@app.route("/api/map-points")
def api_map_points():
    days = config.TIME_FILTERS.get(request.args.get("days", ""))
    include_insp = request.args.get("inspections") == "1" and bool(session.get("authority"))
    points = damage_map.get_report_points(
        severity=request.args.get("severity"), status=request.args.get("status"),
        damage=request.args.get("damage"), days=days, q=request.args.get("q"),
        include_inspections=include_insp)
    return jsonify({"count": len(points), "points": points})


@app.route("/api/reverse-geocode")
def api_reverse_geocode():
    """
    Best-effort GPS -> area/city/state lookup (PART 1). Optional enrichment
    only - the manual Area/City/State fields always work without this, and
    any failure here (offline, timeout, service unavailable) degrades to
    an empty result rather than an error.
    """
    lat = _parse_optional_float(request.args.get("lat"))
    lon = _parse_optional_float(request.args.get("lon"))
    if lat is None or lon is None:
        return {"ok": False, "reason": "missing coordinates"}, 400
    result = geo.reverse_geocode(lat, lon)
    if not result:
        return {"ok": False}
    return {"ok": True, **result}


# ---------------------------------------------------------------------------
# PART 4: Road Segment Health History
# ---------------------------------------------------------------------------

@app.route("/segments")
@authority_required
def segment_list():
    segments = seg.list_segments()
    return render_template("segment_list.html", segments=segments)


@app.route("/road/<segment_id>")
@authority_required
def road_segment_profile(segment_id):
    profile = seg.get_segment_profile(segment_id)
    if not profile:
        flash("No inspections or reports found for that road segment yet.")
        return redirect(url_for("segment_list"))

    chart_w, chart_h, pad = 600, 180, 20
    events = profile["history"]
    svg_points = ""
    svg_dots = []
    if len(events) >= 1:
        n = len(events)
        x_step = (chart_w - 2 * pad) / max(1, n - 1)
        coords = []
        for i, e in enumerate(events):
            x = pad + i * x_step
            y = pad + (chart_h - 2 * pad) * (1 - e["health_score"] / 100.0)
            coords.append((round(x, 1), round(y, 1)))
        svg_points = " ".join(f"{x},{y}" for x, y in coords)
        svg_dots = [{"x": x, "y": y, "score": e["health_score"],
                     "date": time.strftime("%m/%d", time.localtime(e["created_at"]))}
                    for (x, y), e in zip(coords, events)]

    return render_template("road_segment.html", profile=profile, svg_points=svg_points,
                            svg_dots=svg_dots, chart_w=chart_w, chart_h=chart_h)




# ---------------------------------------------------------------------------
# Area view (citizen-friendly) + area list (authority tool)
# ---------------------------------------------------------------------------

@app.route("/area-health")
@authority_required
def area_health_list():
    areas = area_health.list_areas()
    return render_template("area_health_list.html", areas=areas)


@app.route("/area-health/<area_key>")
def area_health_detail(area_key):
    card = area_health.get_area_scorecard(area_key)
    if not card:
        flash("No road reports found for that area yet.")
        return redirect(url_for("road_map"))
    nearby = [r for r in creport.list_all() if geo.area_key(r.area, r.city) == area_key]
    nearby.sort(key=lambda r: r.created_at, reverse=True)
    return render_template("area_health_detail.html", card=card, nearby=nearby[:12])


# ---------------------------------------------------------------------------
# Rewards, profile, prototype certificate
# ---------------------------------------------------------------------------

@app.route("/rewards")
def rewards_page():
    cz = citizens.current_citizen(session)
    summary = rewards.profile_summary(cz.id if cz else None)
    return render_template("rewards.html", summary=summary, history=rewards.history(cz.id if cz else None, 8),
                            eligible=rewards.certificate_eligible(summary["points"]),
                            cert_min=rewards.certificate_min_points())


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        cz = citizens.current_citizen(session, create_if_missing=True)
        citizens.set_name(cz.id, request.form.get("name", ""))
        flash("Profile updated.")
        return redirect(url_for("profile"))
    cz = citizens.current_citizen(session)
    summary = rewards.profile_summary(cz.id if cz else None)
    return render_template("profile.html", summary=summary, history=rewards.history(cz.id if cz else None, 50),
                            eligible=rewards.certificate_eligible(summary["points"]))


@app.route("/rewards/certificate")
def certificate_download():
    cz = citizens.current_citizen(session)
    if not cz or not rewards.certificate_eligible(rewards.total_points(cz.id)):
        flash(f"The certificate unlocks at {rewards.certificate_min_points():,} points.")
        return redirect(url_for("rewards_page"))
    if not cz.name:
        flash("Add your name to your profile so it can appear on the certificate.")
        return redirect(url_for("profile"))
    cert = rewards.issue_certificate(cz)
    return Response(rewards.certificate_pdf(cert), mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={cert['verification_id']}.pdf"})


@app.route("/verify/<verification_id>")
def certificate_verify(verification_id):
    cert = rewards.get_certificate(verification_id.strip().upper())
    return render_template("certificate_verify.html", cert=cert, vid=verification_id), (200 if cert else 404)


# ---------------------------------------------------------------------------
# Legacy URLs (old maintenance dashboard -> authority area)
# ---------------------------------------------------------------------------

@app.route("/maintenance")
@app.route("/maintenance/<path:rest>", methods=["GET", "POST"])
def legacy_maintenance(rest=None):
    return redirect(url_for("authority.dashboard"), code=302)


@app.route("/duplicate-groups/<group_id>")
@authority_required
def duplicate_group_detail(group_id):
    group = dedup.get_duplicate_group(group_id)
    if not group:
        flash(f"No duplicate group found with ID '{group_id}'.")
        return redirect(url_for("authority.analytics"))
    members = creport.list_by_group(group_id)
    return render_template("duplicate_group_detail.html", group=group, members=members)


# ---------------------------------------------------------------------------
# Authority area (separate blueprint: dashboard, assignment, escalation, resolve)
# ---------------------------------------------------------------------------

def _resolve_helpers():
    """Callables authority_views needs from this module (avoids a circular import)."""
    class H:  # simple namespace
        pass
    h = H()
    h.allowed_file, h.save_upload, h.read_image, h.run_pipeline = allowed_file, save_upload, read_image, run_pipeline
    h.RESULT_DIR = RESULT_DIR
    h.explain, h.cmp, h.cv2 = explain, cmp, cv2
    h.DetectorUnavailableError = DetectorUnavailableError
    h.APP_ROOT = APP_ROOT
    return h


register_authority(app, _resolve_helpers())


if __name__ == "__main__":
    print(f"[Roadscan] Detection backend in use: {DETECTOR.name} "
          f"({'PRIMARY - YOLO26' if DETECTOR_STATUS.is_primary else 'FALLBACK - classical CV'})")
    app.run(host="0.0.0.0", port=5000, debug=True)
