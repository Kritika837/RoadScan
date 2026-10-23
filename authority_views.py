"""
authority_views.py
------------------
MODULE: Authority / maintenance area (Flask blueprint, /authority/...)
WHY:    keeps administrative work out of the citizen experience: dashboard,
        assignment, status updates, the 14-day escalation queue, and the
        resolution form where repair evidence (the AFTER photo) is attached.
        The before/after view citizens see is produced here, as part of
        resolving a report - not as a separate technical tool.
PROTOTYPE HONESTY: there is no real authority integration. Sign-in is a
        display name (plus a shared key if ROADSCAN_AUTHORITY_KEY is set in
        the environment). In "open mode" (no key) anyone who knows the URL can
        act as an authority - the layout says so. Departments/officers are
        only what the user types; nothing is pre-filled or invented.
INTERACTION: registered from app.py via register_authority(app, helpers);
        helpers are passed in (detector pipeline, upload saving) so this
        module never imports app.py.
"""

from __future__ import annotations

import os
import time
from functools import wraps

from flask import (Blueprint, render_template, request, redirect, url_for, flash, session)

import area_health
import citizen_report as creport
import config
import dedup
import workflow

AUTH_KEY_ENV = "ROADSCAN_AUTHORITY_KEY"
bp = Blueprint("authority", __name__, url_prefix="/authority")
_H = None  # helpers injected by app.py

PRIORITY_RANK = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# ------------------------------------------------------------------ auth --

def key_required() -> bool:
    return bool(os.environ.get(AUTH_KEY_ENV))


def current_authority():
    a = session.get("authority")
    if a:
        return a
    if not key_required():
        return {"name": "Prototype Officer", "officer_id": None, "open_mode": True}
    return None


def authority_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_authority():
            return redirect(url_for("authority.login", next=request.path))
        return f(*args, **kwargs)
    return wrapper


def _safe_next(target: str) -> str:
    return target if target and target.startswith("/") and not target.startswith("//") else url_for("authority.dashboard")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()[:60]
        if key_required() and request.form.get("key") != os.environ.get(AUTH_KEY_ENV):
            flash("That access key is not correct.")
        elif not name:
            flash("Enter your name so actions can be recorded against you.")
        else:
            session["authority"] = {"name": name, "officer_id": (request.form.get("officer_id") or "").strip()[:40] or None}
            return redirect(_safe_next(request.args.get("next", "")))
    return render_template("authority_login.html", key_required=key_required())


@bp.route("/logout")
def logout():
    session.pop("authority", None)
    flash("Signed out.")
    return redirect(url_for("index"))


def _me() -> str:
    a = current_authority() or {}
    return a.get("name") or "Authority"


# ----------------------------------------------------------------- lists --

def _assigned_to_me(r) -> bool:
    return bool(r.officer_name) and r.officer_name.strip().lower() == _me().strip().lower()


VIEWS = {
    "all": ("All reports", lambda r: True),
    "open": ("Open reports", lambda r: r.status != "RESOLVED"),
    "high": ("High priority", lambda r: r.status != "RESOLVED" and r.repair_priority in ("HIGH", "URGENT")),
    "escalation": ("Escalation required", lambda r: r.is_overdue()),
    "mine": ("Assigned to me", lambda r: r.status != "RESOLVED" and _assigned_to_me(r)),
    "in_progress": ("In progress", lambda r: r.status == "IN_PROGRESS"),
    "resolved": ("Resolved", lambda r: r.status == "RESOLVED"),
}


def _sorted(reports, view):
    if view == "resolved":
        return sorted(reports, key=lambda r: r.resolved_at or 0, reverse=True)
    return sorted(reports, key=lambda r: (not r.is_overdue(), PRIORITY_RANK.get(r.repair_priority, 9), r.created_at))


def _select(view, q=None, severity=None):
    label, fn = VIEWS.get(view, VIEWS["open"])
    rows = [r for r in creport.list_all() if fn(r)]
    if severity:
        rows = [r for r in rows if r.severity_level == severity.upper()]
    if q:
        needle = q.lower()
        rows = [r for r in rows if needle in " ".join(
            filter(None, [r.report_id, r.location, r.area, r.city, r.detected_damage, r.department, r.officer_name])).lower()]
    return label, _sorted(rows, view)


@bp.app_context_processor
def _authority_ctx():
    def citizen_label(r):
        if r.reporter_name:
            return r.reporter_name
        return f"Citizen C-{r.citizen_id:05d}" if r.citizen_id else "Anonymous"
    return {"auth": current_authority(), "auth_key_required": key_required(), "citizen_label": citizen_label,
            "escalation_levels": config.ESCALATION_LEVELS}


@bp.route("/")
@authority_required
def dashboard():
    everything = creport.list_all()
    counts = {k: sum(1 for r in everything if fn(r)) for k, (_l, fn) in VIEWS.items()}
    _, attention = _select("open")
    return render_template("authority_dashboard.html", counts=counts, reports=attention[:12],
                           total_open=counts["open"])


@bp.route("/reports")
@authority_required
def reports():
    view = request.args.get("view", "open")
    label, rows = _select(view, request.args.get("q"), request.args.get("severity"))
    return render_template("authority_reports.html", rows=rows, view=view, label=label, views=VIEWS,
                           q=request.args.get("q", ""), severity=request.args.get("severity", ""))


@bp.route("/assigned")
@authority_required
def assigned():
    label, rows = _select("mine")
    return render_template("authority_reports.html", rows=rows, view="mine", label=label, views=VIEWS, q="", severity="")


@bp.route("/resolved")
@authority_required
def resolved():
    label, rows = _select("resolved")
    return render_template("authority_reports.html", rows=rows, view="resolved", label=label, views=VIEWS, q="", severity="")


@bp.route("/escalations")
@authority_required
def escalations():
    _, rows = _select("escalation")
    rows.sort(key=lambda r: -r.days_overdue)
    return render_template("authority_escalations.html", rows=rows)


@bp.route("/map")
@authority_required
def authority_map():
    return render_template("road_map.html", layout="authority_base.html", authority=True,
                           focus=request.args.get("focus", ""))


@bp.route("/inspect")
@authority_required
def inspect():
    """Technical single-image inspection (the original analyzer), kept for authority staff."""
    return render_template("inspect.html")


@bp.route("/analytics")
@authority_required
def analytics():
    reports_all = creport.list_all()
    done = [r for r in reports_all if r.status == "RESOLVED" and r.resolved_at]
    avg_days = round(sum((r.resolved_at - r.created_at) / 86400 for r in done) / len(done), 1) if done else None
    on_time = round(100 * sum(1 for r in done if r.resolved_at <= r.target_at) / len(done)) if done else None
    sev = {k: sum(1 for r in reports_all if r.severity_level == k) for k in ("CRITICAL", "HIGH", "MODERATE", "LOW")}
    return render_template("authority_analytics.html", stats=creport.stats(), avg_days=avg_days, on_time=on_time,
                           sev=sev, area_overview=area_health.list_areas(),
                           duplicate_groups=dedup.list_duplicate_groups())


# -------------------------------------------------------- one report ------

@bp.route("/reports/<report_id>")
@authority_required
def manage(report_id):
    rep = creport.get_report(report_id)
    if not rep:
        flash(f"No report found with ID '{report_id}'.")
        return redirect(url_for("authority.reports"))
    related = [m for m in creport.list_by_group(rep.duplicate_group_id) if m.report_id != rep.report_id] \
        if rep.duplicate_group_id else []
    return render_template("authority_report.html", rep=rep, timeline=workflow.build_timeline(rep),
                           related=related, history=list(reversed(rep.status_history)),
                           departments=config.DEPARTMENT_SUGGESTIONS,
                           statuses=[s for s in creport.STATUSES if s != "RESOLVED"],
                           today=time.strftime("%Y-%m-%d"))


def _back(report_id):
    return redirect(request.form.get("return_to") if (request.form.get("return_to") or "").startswith("/authority")
                    else url_for("authority.manage", report_id=report_id))


def _flash(result):
    ok, msg = result
    flash(msg)
    return ok


@bp.route("/reports/<report_id>/verify", methods=["POST"])
@authority_required
def verify(report_id):
    _flash(workflow.verify(report_id, _me(), request.form.get("note") or None))
    return _back(report_id)


@bp.route("/reports/<report_id>/assign", methods=["POST"])
@authority_required
def assign(report_id):
    f = request.form
    _flash(workflow.assign(report_id, f.get("department"), f.get("officer_name"), f.get("officer_id"),
                           _me(), (f.get("note") or "").strip() or None))
    return _back(report_id)


@bp.route("/reports/<report_id>/status", methods=["POST"])
@authority_required
def update_status(report_id):
    new_status = request.form.get("status", "")
    if new_status == "RESOLVED":
        flash("To resolve a report, upload the repair evidence (after photo) below.")
        return redirect(url_for("authority.manage", report_id=report_id) + "#resolve")
    _flash(workflow.set_status(report_id, new_status, _me(), (request.form.get("note") or "").strip() or None))
    return _back(report_id)


@bp.route("/reports/<report_id>/escalate", methods=["POST"])
@authority_required
def escalate(report_id):
    _flash(workflow.escalate(report_id, _me(), (request.form.get("note") or "").strip() or None))
    return _back(report_id)


def _parse_date(raw):
    try:
        ts = time.mktime(time.strptime(raw, "%Y-%m-%d")) + 12 * 3600      # noon, so 'today' is never 'future'
    except (TypeError, ValueError):
        return None
    return min(ts, time.time())


def _evidence(rep, after_file):
    """
    Save the AFTER photo and, best-effort, run it through the existing
    pipeline + comparison.py for a measured before/after comparison. The
    comparison only supports the wording 'Visual improvement detected' or
    'Repair evidence submitted' - it never certifies a repair.
    """
    h = _H
    after_path = h.save_upload(after_file)
    url = url_for("static", filename=f"uploads/{os.path.basename(after_path)}")
    annotated_url, comparison = None, None
    try:
        import uuid
        img_after = h.read_image(after_path)
        a_det, a_assess, a_health, _p, a_area = h.run_pipeline(img_after)
        rid = uuid.uuid4().hex
        a_ann = h.explain.draw_annotations(img_after, a_det, {id(a.detection): a.severity for a in a_assess})
        h.cv2.imwrite(os.path.join(h.RESULT_DIR, f"{rid}_after.jpg"), a_ann)
        annotated_url = url_for("static", filename=f"results/{rid}_after.jpg")
        img_before = h.read_image(os.path.join(h.APP_ROOT, rep.image_path.lstrip("/")))
        b_det, b_assess, b_health, _bp, b_area = h.run_pipeline(img_before)
        r = h.cmp.compare_before_after(
            before_health=b_health.score, before_count=len(b_assess), before_area_ratio=b_area,
            after_health=a_health.score, after_count=len(a_assess), after_area_ratio=a_area,
            img_before=img_before, img_after=img_after)
        improved = r.verdict == "Improved" and r.reliable
        comparison = {
            "verdict": r.verdict, "explanation": r.explanation, "health_delta": r.health_delta,
            "count_delta": r.count_delta, "area_delta": r.area_delta, "scene_similarity": r.scene_similarity,
            "low_confidence_comparison": r.low_confidence_comparison, "reliable": r.reliable,
            "after_health_score": a_health.score, "after_condition": a_health.condition,
            "evidence_label": "Visual improvement detected" if improved else "Repair evidence submitted",
        }
    except Exception:
        pass   # the evidence photo is still recorded even if the comparison could not run
    return url, annotated_url, comparison


@bp.route("/reports/<report_id>/resolve", methods=["POST"])
@authority_required
def resolve(report_id):
    rep = creport.get_report(report_id)
    if not rep:
        flash("Report not found.")
        return redirect(url_for("authority.reports"))
    after = request.files.get("after_image")
    if not after or after.filename == "" or not _H.allowed_file(after.filename):
        flash("Repair evidence is required: choose an after-repair photo (JPG or PNG).")
        return redirect(url_for("authority.manage", report_id=report_id) + "#resolve")
    if rep.status == "RESOLVED":
        flash("This report is already resolved.")
        return redirect(url_for("authority.manage", report_id=report_id))
    url, annotated_url, comparison = _evidence(rep, after)
    ok, msg = workflow.resolve(report_id, _me(), url, annotated_url, comparison,
                               (request.form.get("notes") or "").strip()[:1000] or None,
                               _parse_date(request.form.get("resolution_date")))
    flash(msg)
    return redirect(url_for("report_detail", report_id=report_id) if ok
                    else url_for("authority.manage", report_id=report_id) + "#resolve")


def register_authority(app, helpers):
    global _H
    _H = helpers
    app.register_blueprint(bp)
